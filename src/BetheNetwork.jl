module BetheNetwork

export bethe_network, bethe_MPS

using BlockTensors
using BlockTensors.TensorChain
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

function bethe_MPS(network, auxiliary)
    mps = network[begin, :]
    for k in axes(network, 1)[begin + 1 : end]
        mps .*= @view network[k, :]
    end
    auxiliaries_out = only.(matching.(Ref(Outgoing(auxiliary)), network[:, begin]))
    auxiliaries_in = dual.(auxiliaries_out)
    mps[begin] = mergelegs(mps[begin], auxiliaries_out)
    for n in eachindex(mps)[begin + 1 : end - 1]
        mps[n] = mergelegs(mps[n], auxiliaries_out, auxiliaries_in)
    end
    mps[end] = mergelegs(mps[end], auxiliaries_in)
    for t in mps
        BlockTensors.prune!(t)
    end
    return mps
end

end
