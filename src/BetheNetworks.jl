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
export solveBAE, solve_groundstate_BAE
export bethe_energy, heisenberg_hamiltonian_MPO, bethe_network, bethe_MPS

using BlockTensors
using BlockTensors.MatrixProductStates

using TypedTables

function main(
    N, truncation, truncation_transverse, ::Type{S} = Spin;
    optimize = false
) where S <: SymmetrySector
    p = Space("p")
    a = Space("a")
    spectrals = solve_groundstate_BAE(N)
    spectrals_optimalorder = similar(spectrals)
    M = length(spectrals)
    m = div(M, 2, RoundUp)
    spectrals_optimalorder[1:2:M] = spectrals[m:-1:1]
    spectrals_optimalorder[2:2:M] = spectrals[m+1:M]
    network = bethe_network(N, spectrals_optimalorder, p, a, S)
    MPS = bethe_MPS(network, a; truncation...)

    energy = bethe_energy(N, spectrals_optimalorder)
    hamiltonMPO = heisenberg_hamiltonian_MPO(N, p, Space("state"), S)
    original_deviation = abs(real(expectationvalue(hamiltonMPO, MPS) - energy) / energy)
    
    optimize || return original_deviation, MPS

    norms, contractions = optimize_betheMPS!(MPS, network, p, a; truncation_transverse...) 
    deviation = abs(real(expectationvalue(hamiltonMPO, MPS) - energy) / energy)
    
    return original_deviation, deviation, MPS, contractions, norms
end

function entanglement_data(MPS, connecting)
    bonds = 1 : length(MPS) - 1
    dimensions = [bond_dimension(MPS, bond, connecting) for bond in bonds]
    entropies = [entanglement_entropy(MPS, bond, connecting) for bond in bonds]
    return Table(bond = bonds, dimension = dimensions, entropy = entropies)
end

end
