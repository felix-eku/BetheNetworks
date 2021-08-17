module BetheEnergy

export bethe_energy, heisenberg_hamiltonian_MPO

using BlockTensors
using BlockTensors.TensorChain
using ..Pauli


"""
Compute the energy of the Bethe vector for `N` spins with spectral parameters `u`.
"""
function bethe_energy(N::Integer, u::AbstractVector{<:Real})
    return N/4 - sum(1 / (uₙ^2 + 1/4) for uₙ in u)/2
end


function statechange_matrix(
    ::Type{T}, states::Integer, from::Integer, to::Integer
) where T <: Number
    A = zeros(T, states, states)
    A[to, from] = oneunit(T)
    return A
end
statechange_matrix(states, from, to) = statechange_matrix(ComplexF64, states, from, to)

function statechange_tensor(
    ::Type{T}, states::Integer, from::Integer, to::Integer, 
    ::Type{S}, space::Union{Space, Connector}
) where {T <: Number, S <: SymmetrySector}
    dims = SectorDims([zero(S) => states])
    Tensor{T}(
        Dict((zero(S), zero(S)) => statechange_matrix(T, states, from, to)),
        (Leg(Outgoing(space), dims), Leg(Incoming(space), dims)),
        check = false
    )
end
function statechange_tensor(
    ::Type{T}, states::Integer, from::Integer, to::Integer, 
    outgoing::Leg{S, Outgoing}, incoming::Leg{S, Incoming}
) where {T <: Number, S <: SymmetrySector}
    Tensor{T}(
        Dict((zero(S), zero(S)) => statechange_matrix(T, states, from, to)), 
        (outgoing, incoming)
    )
end
function statechange_tensor(states, from, to, space)
    statechange_tensor(ComplexF64, states, from, to, Spin, space)
end
function statechange_tensor(states, from, to, outgoing, incoming)
    statechange_tensor(ComplexF64, states, from, to, outgoing, incoming)
end

function heisenberg_hamiltonian_MPO(
    N::Integer, physical::Union{Space, Connector}, auxiliary::Union{Space, Connector}
)
    t = pauli(0, physical) * (
            statechange_tensor(5, 1, 1, auxiliary) + 
            statechange_tensor(5, 5, 5, auxiliary)
        ) + sum(1:3) do index
        (1/2)pauli(index, physical) * (
            statechange_tensor(5, 1, 1 + index, auxiliary) +
            statechange_tensor(5, 1 + index, 5, auxiliary)
        )
    end
    mpo = chaintensors(n -> (; n), t, auxiliary, N)
    legs = mpo[begin].legs
    p_out = only(matching(Outgoing(physical), legs))
    p_in = only(matching(Incoming(physical), legs))
    a_out = only(matching(Outgoing(auxiliary), legs))
    a_in = only(matching(Incoming(auxiliary), legs))
    mpo[begin] = (
        pauli(0, p_out, p_in) * statechange_tensor(5, 5, 1, a_out, a_in) +
        sum(1:3) do index
            (1/2)pauli(index, p_out, p_in) * (
                statechange_tensor(5, 5, 1 + index, a_out, a_in) +
                statechange_tensor(5, 1 + index, 1, a_out, a_in)
            )
        end
    )
    return mpo
end

end
