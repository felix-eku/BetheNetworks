module BetheNetwork

export bethe_network, bethe_MPS, optimize_betheMPS!

using BlockTensors
using BlockTensors.TensorChain
using BlockTensors.MatrixProductStates
using ..Pauli
using ..LaxOperators

function bethe_network(N, spectral_parameters, physical, auxiliary)
    operators = convert(
        Vector{Tensor{ComplexF64, Spin}},
        lax_operators(spectral_parameters, physical, auxiliary)
    )
    network = chaintensors(n -> (; n), operators, auxiliary, N)
    for k in axes(network, 1)
        network[k, 1] = network[k, 1] * Tensor(
            Dict(tuple(Spin(Sᶻ = -1)) => ones(ComplexF64, 1)),
            dual(only(matching(Incoming(auxiliary), network[k, 1])), connect = true)
        )
        network[k, N] = network[k, N] * Tensor(
            Dict(tuple(Spin(Sᶻ = 1)) => ones(ComplexF64, 1)),
            dual(only(matching(Outgoing(auxiliary), network[k, N])), connect = true)
        )
    end
    for n in axes(network, 2)
        network[1, n] = network[1, n] * Tensor(
            Dict(tuple(Spin(Sᶻ = 1)) => ones(ComplexF64, 1)),
            dual(only(matching(Incoming(physical), network[1, n])), connect = true)
        )
    end
    return network
end

function bethe_MPS(network, auxiliary, params...)
    MPS = network[begin, :]
    for k in axes(network, 1)[begin + 1 : end]
        MPS = MPO_MPS_contraction(view(network, k, :), MPS, auxiliary)
        canonicalize!(MPS, eachindex(MPS), Outgoing(auxiliary))
        if !isempty(params)
            canonicalize!(MPS, reverse(eachindex(MPS)), Incoming(auxiliary), params...)
        end
    end
    return MPS
end

function optimize_betheMPS!(MPS, network, physical, auxiliary, maxblockdim)
    @assert size(network, 2) == length(MPS)
    scaling = ones(float(real(eltype(eltype(MPS)))), size(network, 2))
    contractions = similar(network, size(network, 1) + 1, size(network, 2))
    N = axes(network, 2)
    Nbegin = N[begin : end - 2]
    Nmiddle = N[begin + 1 : end - 1]
    Nend = N[begin + 2 : end]
    MPS[Nbegin[begin]], MPS[Nmiddle[begin]] = exchangegauge(
        MPS[Nbegin[begin]], MPS[Nmiddle[begin]], Outgoing(auxiliary)
    )
    contractions[axes(network, 1), [begin, end]] .= network[:, [begin, end]]
    contractions[end, begin] = MPS[begin]'
    for ns in zip(Nbegin, Nmiddle, Nend)
        updatelocally!(
            MPS, contractions, scaling, network, 
            physical, Outgoing(auxiliary), maxblockdim, ns...
        ) do MPS, contractions, auxiliary, nprev, n, nnext
            MPS[n], MPS[nnext] = exchangegauge(MPS[n], MPS[nnext], auxiliary)
            nothing
        end
    end
    norm = updatelocally!(
        MPS, contractions, Incoming(auxiliary), Nmiddle[end], Nend[end]
    )
    norm *= scaling[Nmiddle[end]]
    @show norm; norms = [norm]
    sizehint!(norms, 2 * length(MPS))
    while length(norms) < length(MPS) || !isapprox(norms[end], norms[end - length(MPS) + 1])
        for (nprev, n, nnext) in zip(reverse(Nend), reverse(Nmiddle), reverse(Nbegin)) 
            norm = updatelocally!(
                optimize!, MPS, contractions, scaling, network, 
                physical, Incoming(auxiliary), maxblockdim, nprev, n, nnext
            )
            norm *= scaling[nprev] * scaling[nnext]
            @show norm; push!(norms, norm)
        end
        norm = updatelocally!(
            MPS, contractions, Outgoing(auxiliary), Nmiddle[begin], Nbegin[begin]
        )
        norm *= scaling[Nmiddle[begin]]
        @show norm; push!(norms, norm)
        for (nprev, n, nnext) in zip(Nbegin, Nmiddle, Nend)
            norm = updatelocally!(
                optimize!, MPS, contractions, scaling, network, 
                physical, Outgoing(auxiliary), maxblockdim, nprev, n, nnext
            )
            norm *= scaling[nprev] * scaling[nnext]
            @show norm; push!(norms, norm)
        end
        norm = updatelocally!(
            MPS, contractions, Incoming(auxiliary), Nmiddle[end], Nend[end]
        )
        norm *= scaling[Nmiddle[end]]
        @show norm; push!(norms, norm)
        if length(norms) > 9 * length(MPS) break end
    end
    return norms, contractions
end

function optimize!(MPS, contractions, auxiliary, nprev, n, nnext)
    contraction = 1
    for k in axes(contractions, 1)[begin : end - 1]
        contraction = (contractions[k, n] * contraction) * contractions[k, nnext]
    end
    MPS[n], R = qr(
        (contractions[end, nprev] * contraction) * contractions[end, nnext], 
        auxiliary
    )
    norm = √real(R'R)
    MPS[nnext] = R / norm * MPS[nnext]
    return norm
end

function updatelocally!(
    update!, MPS, contractions, scaling, network, 
    physical, auxiliary, maxblockdim, nprev, n, nnext
)
    Knet = axes(network, 1)
    contractions[Knet, n] .= view(contractions, Knet, nprev) .* view(network, :, n)
    result = update!(MPS, contractions, auxiliary, nprev, n, nnext)
    contractions[end, n] = contractions[end, nprev] * MPS[n]'
    K = axes(contractions, 1)
    for k in K 
        contractions[k, n] = mergelegs(
            contractions[k, n],
            union(
                matching(Outgoing(physical), contractions[k, nprev]),
                matching(Outgoing(physical), contractions[k, n])
            ),
            union(
                matching(Incoming(physical), contractions[k, nprev]),
                matching(Incoming(physical), contractions[k, n])
            )
        )
    end
    canonicalize!(view(contractions, :, n), K, Outgoing(physical), normalize = false)
    norm = canonicalize!(
        view(contractions, :, n), reverse(K), Incoming(physical), maxblockdim,
        normalize = true
    )
    scaling[n] = scaling[nprev] * norm
    return result
end

function updatelocally!(
    MPS, contractions, auxiliary, nprev, n
)
    contraction = 1
    for k in axes(contractions, 1)[begin : end - 1]
        contraction = contractions[k, nprev] * (contraction * contractions[k, n])
    end
    MPS[n], R = qr(contractions[end, nprev] * contraction, auxiliary)
    norm = √real(R'R)
    MPS[nprev] = MPS[nprev] * R / norm
    contractions[end, n] = MPS[n]'
    return norm
end


end
