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
    states::Integer, from::Integer, to::Integer, ::Type{T} = ComplexF64
) where T <: Number
    A = zeros(T, states, states)
    A[to, from] = oneunit(T)
    return A
end

function statechange_tensor(
    states::Integer, from::Integer, to::Integer, space::Union{Space, Connector},
    ::Type{T} = ComplexF64, ::Type{S} = Spin
) where {T <: Number, S <: SymmetrySector}
    dims = SectorDims([zero(S) => states])
    Tensor{T}(
        Dict((zero(S), zero(S)) => statechange_matrix(states, from, to, T)),
        (Leg(Outgoing(space), dims), Leg(Incoming(space), dims)),
        check = false
    )
end
function statechange_tensor(
    states, from, to, space::Union{Space, Connector}, ::Type{S}
) where S <: SymmetrySector
    statechange_tensor(states, from, to, space, ComplexF64, S)
end
function statechange_tensor(
    states::Integer, from::Integer, to::Integer, 
    outgoing::Leg{S, Outgoing}, incoming::Leg{S, Incoming},
    ::Type{T} = ComplexF64, ::Type{S} = Spin,
) where {T <: Number, S <: SymmetrySector}
    Tensor{T}(
        Dict((zero(S), zero(S)) => statechange_matrix(states, from, to, T)), 
        (outgoing, incoming)
    )
end
function statechange_tensor(
    states, from, to, 
    outgoing::Leg{S, Outgoing}, incoming::Leg{S, Incoming}, 
    ::Type{S}
) where S <: SymmetrySector
    statechange_tensor(states, from, to, outgoing, incoming, ComplexF64, S)
end

function heisenberg_hamiltonian_MPO(
    N::Integer, 
    physical::Union{Space, Connector}, 
    auxiliary::Union{Space, Connector}, 
    types::Type...
)
    t = pauli(0, physical, types...) * (
            statechange_tensor(5, 1, 1, auxiliary, types...) + 
            statechange_tensor(5, 5, 5, auxiliary, types...)
        ) + sum(1:3) do index
            (1/2)pauli(index, physical, types...) * (
                statechange_tensor(5, 1, 1 + index, auxiliary, types...) +
                statechange_tensor(5, 1 + index, 5, auxiliary, types...)
            )
        end
    mpo = chaintensors(n -> (; n), t, auxiliary, N)
    legs = mpo[begin].legs
    p_out = only(matching(Outgoing(physical), legs))
    p_in = only(matching(Incoming(physical), legs))
    a_out = only(matching(Outgoing(auxiliary), legs))
    a_in = only(matching(Incoming(auxiliary), legs))
    mpo[begin] = (
        pauli(0, p_out, p_in, types...) * statechange_tensor(5, 5, 1, a_out, a_in, types...) +
        sum(1:3) do index
            (1/2)pauli(index, p_out, p_in, types...) * (
                statechange_tensor(5, 5, 1 + index, a_out, a_in, types...) +
                statechange_tensor(5, 1 + index, 1, a_out, a_in, types...)
            )
        end
    )
    return mpo
end

end
