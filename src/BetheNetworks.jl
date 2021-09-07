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

function main(N, maxdim_initial, maxdim)
    p = Space("p")
    a = Space("a")
    spectrals = solve_groundstate_BAE(N)
    network = bethe_network(N, spectrals, p, a)
    MPS = bethe_MPS(network, a, maxdim_initial)
    norms = optimize_betheMPS!(MPS, network, p, a, maxdim)
    
    energy = bethe_energy(N, spectrals)
    hamiltonMPO = heisenberg_hamiltonian_MPO(N, p, Space("state"))
    deviation = abs(real(expectationvalue(hamiltonMPO, MPS) - energy) / energy)
    return deviation, MPS, norms
end

end
