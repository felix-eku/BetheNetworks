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

function main(N, maxdim_initial, maxdim)
    p = Space("p")
    a = Space("a")
    spectrals = solve_groundstate_BAE(N)
    network = bethe_network(N, spectrals, p, a)
    MPS = bethe_MPS(network, a, maxdim_initial)

    energy = bethe_energy(N, spectrals)
    hamiltonMPO = heisenberg_hamiltonian_MPO(N, p, Space("state"))
    original_deviation = abs(real(expectationvalue(hamiltonMPO, MPS) - energy) / energy)
    
    norms, contractions = optimize_betheMPS!(MPS, network, p, a, maxdim) 
    deviation = abs(real(expectationvalue(hamiltonMPO, MPS) - energy) / energy)
    
    return original_deviation, deviation, MPS, contractions, norms
end

function entanglement_data(MPS, connecting)
    bonds = 1 : length(MPS) - 1
    dimensions = [
        only(matching(connecting, MPS[eachindex(MPS)[bond]])).dimensions.totaldim
        for bond in bonds
    ]
    entropies = [entanglement_entropy(MPS, bond, connecting) for bond in bonds]
    return Table(bond = bonds, dimension = dimensions, entropy = entropies)
end

end
