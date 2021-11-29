module LaxOperators

using BlockTensors
using BlockTensors.TensorChain
using ..Pauli

export lax_operator, lax_operators

function lax_operator(
    u, physical::Union{Space, Connector}, auxiliary::Union{Space, Connector}, types...
)
    phys = copy(physical)
    aux = copy(auxiliary)
    addtags!(aux; u = round(u, digits = 3))
    lax_operator(u, Outgoing(phys), Incoming(phys), Outgoing(aux), Incoming(aux), types...)
end

function lax_operator(
    u, physout::Union{Leg{S, Outgoing}, Outgoing}, physin::Union{Leg{S, Incoming}, Incoming},
    auxout::Union{Leg{S, Outgoing}, Outgoing}, auxin::Union{Leg{S, Incoming}, Incoming}, 
    types...
) where S <: SymmetrySector
    scaling = u * pauli(0, physout, physin, types...) * pauli(0, auxout, auxin, types...)
    return scaling + (im/2) * sum(1:3) do index
        pauli(index, physout, physin, types...) * pauli(index, auxout, auxin, types...)
    end
end

function lax_operators(spectral_parameters, physical, auxiliary, types...)
    operators = [
        lax_operator(u, physical, auxiliary, types...)
        for u in spectral_parameters
    ]
    connectchain!(operators, physical)
    return operators
end

end
