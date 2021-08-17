module Pauli

using BlockTensors

export Spin, pauli

@SymmetrySector Spin {Sᶻ}

function pauli(
    ::Type{T}, index::Union{Integer, AbstractString}, 
    outgoing::Union{Leg{Spin, Outgoing}, Outgoing}, 
    incoming::Union{Leg{Spin, Incoming}, Incoming}
) where T <: Number
    Tensor{T}(paulicomponents(T, index), (outgoing, incoming))
end
function pauli(
    ::Type{T}, index::Union{Integer, AbstractString}, 
    space::Union{Connector, Space}
) where T <: Number
    dims = SectorDims([Spin(Sᶻ = -1) => 1, Spin(Sᶻ = +1) => 1])
    Tensor{T}(
        paulicomponents(T, index), 
        (Leg(Outgoing(space), dims), Leg(Incoming(space), dims)),
        check = false
    )
end
pauli(index::Union{Integer, AbstractString}, legs...) = pauli(ComplexF64, index, legs...)

function paulicomponents(::Type{T}, index::Integer) where T <: Number
    index == 0 && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = +1)) => ones(T, 1, 1),
        (Spin(Sᶻ = -1), Spin(Sᶻ = -1)) => ones(T, 1, 1)
    )
    index == 1 && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = -1)) => ones(T, 1, 1),
        (Spin(Sᶻ = -1), Spin(Sᶻ = +1)) => ones(T, 1, 1)
    )
    index == 2 && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = -1)) => fill(-one(T)im, 1, 1),
        (Spin(Sᶻ = -1), Spin(Sᶻ = +1)) => fill(+one(T)im, 1, 1)
    )
    index == 3 && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = +1)) => fill(+one(T), 1, 1),
        (Spin(Sᶻ = -1), Spin(Sᶻ = -1)) => fill(-one(T), 1, 1)
    )
    throw(ArgumentError("index $index ∉ (0, 1, 2, 3)"))
end
function paulicomponents(::Type{T}, index::AbstractString) where T <: Number
    index == "x" && return paulicomponents(T, 1)
    index == "y" && return paulicomponents(T, 2)
    index == "z" && return paulicomponents(T, 3)
    index == "+" && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = -1)) => ones(T, 1, 1)
    )
    index == "-" && return Dict(
        (Spin(Sᶻ = -1), Spin(Sᶻ = +1)) => ones(T, 1, 1)
    )
    throw(ArgumentError("unknown index $index"))
end

end
