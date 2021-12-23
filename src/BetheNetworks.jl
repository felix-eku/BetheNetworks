module BetheNetworks

include("Pauli.jl")
include("LaxOperators.jl")
include("SpectralParameters.jl")
include("BetheEnergy.jl")
include("BetheNetwork.jl")

using .Pauli
using .LaxOperators

using .SpectralParameters
using .BetheEnergy
using .BetheNetwork

export Spin, pauli, lax_operator, lax_operators
export solveBAE, groundstate_Bₙ
export bethe_energy, heisenberg_hamiltonian_MPO, bethe_network, betheMPS
export study_betheMPS

using BlockTensors
using BlockTensors.MatrixProductStates

using TypedTables

function study_betheMPS(
    N, Bₙ, truncation, truncation_transverse, ::Type{S} = Spin;
    real_symmetric = false, optimize = false, bond = nothing
) where S <: SymmetrySector
    p = Space("p")
    a = Space("a")
    spectrals = solveBAE(N, Bₙ)
    m, r = divrem(length(spectrals), 2, RoundUp)
    if real_symmetric
        MPOs = symMPO.(N, spectrals[m + 1 : end], p, a, S)
        if r < 0
            @assert isapprox(spectrals[m], 0, atol = 4eps())
            MPO0 = B_MPO(N, 0.0, p, a, S)
            for n in LinearIndices(MPO0)
                MPO0[n] *= (-1)^n * im
            end
            pushfirst!(MPOs, MPO0)
        end
        network = bethe_network(MPOs, p)
    else
        spectrals_optimalorder = similar(spectrals)
        spectrals_optimalorder[1:2:end] = spectrals[m : -1 : begin]
        spectrals_optimalorder[2:2:end] = spectrals[m + 1 : end]
        network = bethe_network(N, spectrals_optimalorder, p, a, S)
    end
    MPS, entanglements = betheMPS(network, a, bond; truncation...)

    energy = bethe_energy(N, spectrals)
    hamiltonMPO = heisenberg_hamiltonian_MPO(N, p, Space("state"), S)
    original_deviation = abs(real(expectationvalue(hamiltonMPO, MPS, p) - energy) / energy)
    
    optimize || return original_deviation, MPS, entanglements

    norms, contractions = optimize_betheMPS!(MPS, network, p, a; truncation_transverse...) 
    deviation = abs(real(expectationvalue(hamiltonMPO, MPS, p) - energy) / energy)
    
    return original_deviation, deviation, MPS, entanglements, contractions, norms
end

function entanglement_data(MPS, connecting)
    bonds = 1 : length(MPS) - 1
    dimensions = [bond_dimension(MPS, bond, connecting) for bond in bonds]
    entropies = [entanglement_entropy(MPS, bond, connecting) for bond in bonds]
    return Table(bond = bonds, dimension = dimensions, entropy = entropies)
end

function truncation_comparison(N, Bₙ, maxrelerror)
    deviation_relerror, MPS = study_betheMPS(N, Bₙ, (; maxrelerror), ())
    bond_dimension = bond_dimension(MPS, div(N, 2), Outgoing(:a))
    deviation_maxdim, = study_betheMPS(N, Bₙ, (; maxdim = bond_dimension), ())
    return bond_dimension, deviation_relerror, deviation_maxdim
end
end
