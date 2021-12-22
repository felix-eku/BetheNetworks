module BetheNetwork

export bethe_network, bethe_MPS, optimize_betheMPS!

using BlockTensors
using BlockTensors.TensorChain
using BlockTensors.MatrixProductStates
using ..Pauli
using ..LaxOperators


function B_MPO(N::Integer, lax::Tensor, auxiliary, types...)
    MPO = createchain(lax, N) do n, connector
        matching(auxiliary, connector) ? () : (; n)
    end
    connectchain!(MPO, auxiliary)
    T, S = eltype(MPO).parameters
    MPO = convert(Vector{Tensor{T,S}}, MPO)
    a_in = only(matching(Incoming(auxiliary), MPO[begin]))
    MPO[begin] = MPO[begin] * spindown(dual(a_in, connect = true), types...)
    a_out = only(matching(Outgoing(auxiliary), MPO[end]))
    MPO[end] = MPO[end] * spinup(dual(a_out, connect = true), types...)
    return MPO
end
function B_MPO(N::Integer, u::Number, physical, auxiliary, types...)
    B_MPO(N, lax_operator(u, physical, auxiliary, types...), auxiliary, types...)
end
function bethe_network(N::Integer, spectral_parameters, physical, auxiliary, types...)
    operators = lax_operators(spectral_parameters, physical, auxiliary, types...)
    bethe_network(B_MPO.(N, operators, auxiliary, types), physical)
end
function bethe_network(MPOs, physical)
    network = cat(reshape.(MPOs, 1, :)..., dims = 1)
    T, S = eltype(network).body.parameters
    connectchain!(network, physical, dim = 1)
    for n in axes(network, 2)
        p = only(matching(Incoming(physical), network[begin, n]))
        network[begin, n] = network[begin, n] * spinup(dual(p, connect = true), T, S)
    end
    return network
end

function bethe_MPS(network, auxiliary; truncation...)
    MPS = network[begin, :]
    for k in axes(network, 1)[begin + 1 : end]
        MPS = MPO_MPS_contraction(view(network, k, :), MPS, auxiliary)
        canonicalize!(MPS, eachindex(MPS), Outgoing(auxiliary))
        if !isempty(truncation)
            canonicalize!(MPS, reverse(eachindex(MPS)), Incoming(auxiliary); truncation...)
        end
    end
    return MPS
end

function optimize_betheMPS!(MPS, network, physical, auxiliary; truncation...)
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
            physical, Outgoing(auxiliary), ns...; 
            truncation...
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
                physical, Incoming(auxiliary), nprev, n, nnext;
                truncation...
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
                physical, Outgoing(auxiliary), nprev, n, nnext;
                truncation...
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
    physical, auxiliary, nprev, n, nnext; truncation...
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
    canonicalize!(
        view(contractions, :, n), reverse(K), Incoming(physical), normalize = false
    )
    norm = canonicalize!(
        view(contractions, :, n), K, Outgoing(physical), normalize = true; truncation...
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
