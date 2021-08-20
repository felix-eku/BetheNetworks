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
    contractions = similar(network, size(network, 1) + 1, size(network, 2))
    K = axes(network, 2)
    Kbegin = K[begin : end - 2]
    Kmiddle = K[begin + 1 : end - 1]
    Kend = K[begin + 2 : end]
    MPS[Kbegin[begin]], MPS[Kmiddle[begin]] = exchangegauge(
        MPS[Kbegin[begin]], MPS[Kmiddle[begin]], Outgoing(auxiliary)
    )
    contractions[axes(network, 1), [begin, end]] .= network[:, [begin, end]]
    contractions[end, begin] = MPS[begin]'
    for ks in zip(Kbegin, Kmiddle, Kend)
        updatelocally!(
            MPS, contractions, network, 
            physical, Outgoing(auxiliary), maxblockdim, ks...
        ) do MPS, contractions, auxiliary, kprev, k, knext
            MPS[k], MPS[knext] = exchangegauge(MPS[k], MPS[knext], auxiliary)
            nothing
        end
    end
    norm² = updatelocally!(
        MPS, contractions, Incoming(auxiliary), Kmiddle[end], Kend[end]
    )
    @show norm²; norm²s = [norm²]
    sizehint!(norm²s, 2 * length(MPS))
    while length(norm²s) < length(MPS) || !isapprox(norm²s[end], norm²s[end - length(MPS)])
        for ks in zip(reverse(Kend), reverse(Kmiddle), reverse(Kbegin)) 
            norm² = updatelocally!(
                optimize!, MPS, contractions, network, 
                physical, Incoming(auxiliary), maxblockdim, ks...
            )
            @show norm²; push!(norm²s, norm²)
        end
        norm² = updatelocally!(
            MPS, contractions, Outgoing(auxiliary), Kmiddle[begin], Kbegin[begin]
        )
        @show norm²; push!(norm²s, norm²)
        for ks in zip(Kbegin, Kmiddle, Kend)
            norm² = updatelocally!(
                optimize!, MPS, contractions, network, 
                physical, Outgoing(auxiliary), maxblockdim, ks...
            )
            @show norm²; push!(norm²s, norm²)
        end
        norm² = updatelocally!(
            MPS, contractions, Incoming(auxiliary), Kmiddle[end], Kend[end]
        )
        @show norm²; push!(norm²s, norm²)
    end
    return norm²s
end

function optimize!(MPS, contractions, auxiliary, kprev, k, knext)
    contraction = 1
    for l in axes(contractions, 1)[begin : end - 1]
        contraction = (contractions[l, k] * contraction) * contractions[l, knext]
    end
    MPS[k], R = qr(
        (contractions[end, kprev] * contraction) * contractions[end, knext], 
        auxiliary
    )
    MPS[knext] = R * MPS[knext]
    return real(R'R)
end

function updatelocally!(
    update!, MPS, contractions, network, 
    physical, auxiliary, maxblockdim, kprev, k, knext
)
    Lnet = axes(network, 1)
    contractions[Lnet, k] .= view(contractions, Lnet, kprev) .* view(network, :, k)
    result = update!(MPS, contractions, auxiliary, kprev, k, knext)
    contractions[end, k] = contractions[end, kprev] * MPS[k]'
    L = axes(contractions, 1)
    for l in L 
        contractions[l, k] = mergelegs(
            contractions[l, k],
            union(
                matching(Outgoing(physical), contractions[l, kprev]),
                matching(Outgoing(physical), contractions[l, k])
            ),
            union(
                matching(Incoming(physical), contractions[l, kprev]),
                matching(Incoming(physical), contractions[l, k])
            )
        )
    end
    canonicalize!(view(contractions, :, k), L, Outgoing(physical), normalize = false)
    canonicalize!(
        view(contractions, :, k), reverse(L), Incoming(physical), maxblockdim,
        normalize = false
    )
    return result
end

function updatelocally!(
    MPS, contractions, auxiliary, kprev, k
)
    contraction = 1
    for l in axes(contractions, 1)[begin : end - 1]
        contraction = contractions[l, kprev] * (contraction * contractions[l, k])
    end
    MPS[k], R = qr(contractions[end, kprev] * contraction, auxiliary)
    MPS[kprev] = MPS[kprev] * R
    contractions[end, k] = MPS[k]'
    return real(R'R)
end


end
