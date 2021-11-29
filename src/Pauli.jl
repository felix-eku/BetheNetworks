module Pauli

using BlockTensors

export Spin, spinup, spindown, pauli

@SymmetrySector Spin {Sᶻ}

function spinup(
    leg::Union{Leg{S}, Connector},
    ::Type{T} = ComplexF64, ::Type{S} = Spin, 
    direction::AbstractChar = 'z'
) where {T <: Number, S <: SymmetrySector}
    Tensor{T}(spinupcomponents(T, S, direction), leg)
end
function spinup(leg, ::Type{S}, direction::AbstractChar = 'z') where S <: SymmetrySector
    spinup(leg, ComplexF64, S, direction)
end
spinup(leg, direction::AbstractChar) = spinup(leg, ComplexF64, Spin, direction)

function spinupcomponents(
    ::Type{T}, ::Type{Spin}, direction::AbstractChar
) where T <: Number
    direction == 'x' && return Dict(
        tuple(Spin(Sᶻ = +1) => ones(T, 1)),
        tuple(Spin(Sᶻ = -1) => ones(T, 1))
    )
    direction == 'y' && return Dict(
        tuple(Spin(Sᶻ = +1) => ones(T, 1)),
        tuple(Spin(Sᶻ = -1) => ones(T, 1)im)
    )
    direction == 'z' && return Dict(
        tuple(Spin(Sᶻ = 1)) => ones(T, 1)
    )
    throw(ArgumentError("unknown index $index"))
end

function spinupcomponents(
    ::Type{T}, ::Type{Trivial}, direction::AbstractChar
) where T <: Number
    direction == 'x' && return Dict(
        tuple(Trivial()) => ones(T, 2)
    )
    direction == 'y' && return Dict(
        tuple(Trivial()) => T[one(T), one(T)im]
    )
    direction == 'z' && return Dict(
        tuple(Trivial()) => T[one(T), zero(T)]
    )
    throw(ArgumentError("unknown index $index"))
end

function spindown(
    leg::Union{Leg{S}, Connector},
    ::Type{T} = ComplexF64, ::Type{S} = Spin, 
    direction::AbstractChar = 'z'
) where {T <: Number, S <: SymmetrySector}
    Tensor{T}(spindowncomponents(T, S, direction), leg)
end
function spindown(leg, ::Type{S}, direction::AbstractChar = 'z') where S <: SymmetrySector
    spindown(leg, ComplexF64, S, direction)
end
spindown(leg, direction::AbstractChar) = spindown(leg, ComplexF64, Spin, direction)

function spindowncomponents(
    ::Type{T}, ::Type{Spin}, direction::AbstractChar
) where T <: Number
    direction == 'x' && return Dict(
        tuple(Spin(Sᶻ = +1) => ones(T, 1)),
        tuple(Spin(Sᶻ = -1) => -ones(T, 1))
    )
    direction == 'y' && return Dict(
        tuple(Spin(Sᶻ = +1) => ones(T, 1)),
        tuple(Spin(Sᶻ = -1) => -ones(T, 1)im)
    )
    direction == 'z' && return Dict(
        tuple(Spin(Sᶻ = -1)) => ones(T, 1)
    )
    throw(ArgumentError("unknown index $index"))
end

function spindowncomponents(
    ::Type{T}, ::Type{Trivial}, direction::AbstractChar
) where T <: Number
    direction == 'x' && return Dict(
        tuple(Trivial()) => T[one(T), -one(T)]
    )
    direction == 'y' && return Dict(
        tuple(Trivial()) => T[one(T), -one(T)im]
    )
    direction == 'z' && return Dict(
        tuple(Trivial()) => T[zero(T), one(T)]
    )
    throw(ArgumentError("unknown index $index"))
end

function pauli(
    index::Union{Integer, AbstractChar}, 
    outgoing::Union{Leg{S, Outgoing}, Outgoing}, 
    incoming::Union{Leg{S, Incoming}, Incoming},
    ::Type{T} = ComplexF64, ::Type{S} = Spin
) where {T <: Number, S <: SymmetrySector}
    Tensor{T}(paulicomponents(T, S, index), (outgoing, incoming), fill(paulidims(S), 2))
end
function pauli(
    index::Union{Integer, AbstractChar}, 
    outgoing::Union{Leg{S, Outgoing}, Outgoing}, 
    incoming::Union{Leg{S, Incoming}, Incoming},
    ::Type{S}
) where S <: SymmetrySector
    pauli(index, outgoing, incoming, ComplexF64, S)
end
function pauli(
    index::Union{Integer, AbstractChar},
    space::Union{Connector, Space},
    ::Type{T} = ComplexF64, ::Type{S} = Spin
) where {T <: Number, S <: SymmetrySector}
    pauli(index, Outgoing(space), Incoming(space), T, S)
end
function pauli(
    index::Union{Integer, AbstractChar}, space::Union{Connector, Space}, ::Type{S}
) where S <: SymmetrySector
    pauli(index, space, ComplexF64, S)
end

paulidims(::Type{Trivial}) = SectorDims([Trivial() => 2])
paulidims(::Type{Spin}) = SectorDims([Spin(Sᶻ = -1) => 1, Spin(Sᶻ = +1) => 1])

function paulicomponents(::Type{T}, ::Type{Spin}, index::Integer) where T <: Number
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
function paulicomponents(::Type{T}, ::Type{Spin}, index::AbstractChar) where T <: Number
    index == 'x' && return paulicomponents(T, Spin, 1)
    index == 'y' && return paulicomponents(T, Spin, 2)
    index == 'z' && return paulicomponents(T, Spin, 3)
    index == '+' && return Dict(
        (Spin(Sᶻ = +1), Spin(Sᶻ = -1)) => ones(T, 1, 1)
    )
    index == '-' && return Dict(
        (Spin(Sᶻ = -1), Spin(Sᶻ = +1)) => ones(T, 1, 1)
    )
    throw(ArgumentError("unknown index $index"))
end

function paulicomponents(::Type{T}, ::Type{Trivial}, index::Integer) where T <: Number
    index == 0 && return Dict(
        (Trivial(), Trivial()) => T[one(T) zero(T); zero(T) one(T)]
    )
    index == 1 && return Dict(
        (Trivial(), Trivial()) => T[zero(T) one(T); one(T) zero(T)]
    )
    index == 2 && return Dict(
        (Trivial(), Trivial()) => T[zero(T) -one(T)im; +one(T)im zero(T)]
    )
    index == 3 && return Dict(
        (Trivial(), Trivial()) => T[one(T) zero(T); zero(T) -one(T)]
    )
    throw(ArgumentError("index $index ∉ (0, 1, 2, 3)"))
end
function paulicomponents(::Type{T}, ::Type{Trivial}, index::AbstractChar) where T <: Number
    index == 'x' && return paulicomponents(T, Trivial, 1)
    index == 'y' && return paulicomponents(T, Trivial, 2)
    index == 'z' && return paulicomponents(T, Trivial, 3)
    index == '+' && return Dict(
        (Trivial(), Trivial()) => T[zero(T) one(T); zero(T) zero(T)]
    )
    index == '-' && return Dict(
        (Trivial(), Trivial()) => T[zero(T) zero(T); one(T) zero(T)]
    )
    throw(ArgumentError("unknown index $index"))
end

end
