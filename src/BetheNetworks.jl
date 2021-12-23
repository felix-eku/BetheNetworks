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
export study_betheMPS, study_betheMPS_projection

using BlockTensors
using BlockTensors.MatrixProductStates

using TypedTables, CSV

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

function study_betheMPS_projection(
    N, Bₙ, ::Type{S} = Spin; truncation...
) where S <: SymmetrySector
    p = Space("p")
    a = Space("a")
    spectrals = solveBAE(N, Bₙ)
    m = div(length(spectrals), 2, RoundUp)
    spectrals_optimalorder = similar(spectrals)
    spectrals_optimalorder[1:2:end] = spectrals[m : -1 : begin]
    spectrals_optimalorder[2:2:end] = spectrals[m + 1 : end]
    network = bethe_network(N, spectrals_optimalorder, p, a, S)
    MPS = betheMPS_projection(network, p, a; truncation...)
    canonicalize!(MPS, eachindex(MPS), Incoming(a))

    energy = bethe_energy(N, spectrals)
    hamiltonMPO = heisenberg_hamiltonian_MPO(N, p, Space("state"), S)
    deviation = abs(real(expectationvalue(hamiltonMPO, MPS, p) - energy) / energy)
    return deviation, MPS
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

function main_original_no_blocks()
    Ns = 16:2:36
    deviations = Array{Float64}(undef, length(Ns), 6)
    bond_dims = similar(deviations, Int)
    for (i, N) in enumerate(Ns)
        Bₙs = [groundstate_Bₙ(N), -N/2 + 2 : 2 : +N/2 - 2]
        truncations = [(; maxdim = 20N), (; maxdim = 30N), (; maxrelerror = 1e-9)]
        for (configuration, (truncation, Bₙ)) in enumerate(Iterators.product(truncations, Bₙs))
            deviation, MPS = study_betheMPS(N, Bₙ, truncation, (), Trivial)
            bond_dim = maximum(bond_dimension(MPS, bond, Outgoing(:a)) for bond = 1:N-1)
            @show (N, deviation, bond_dim)
            deviations[i, configuration] = deviation
            bond_dims[i, configuration] = bond_dim
        end
    end
    table = Table(
        N = Ns,
        deviation_ground_state_maxdim_20N = deviations[:, 1],
        deviation_ground_state_maxdim_30N = deviations[:, 2],
        deviation_ground_state_truncation_error_e9 = deviations[:, 3],
        deviation_excited_state_maxdim_20N = deviations[:, 4],
        deviation_excited_state_maxdim_30N = deviations[:, 5],
        deviation_excited_state_truncation_error_e9 = deviations[:, 6],
        bond_dim_ground_state_maxdim_20N = bond_dims[:, 1],
        bond_dim_ground_state_maxdim_30N = bond_dims[:, 2],
        bond_dim_ground_state_truncation_error_e9 = bond_dims[:, 3],
        bond_dim_excited_state_maxdim_20N = bond_dims[:, 4],
        bond_dim_excited_state_maxdim_30N = bond_dims[:, 5],
        bond_dim_excited_state_truncation_error_e9 = bond_dims[:, 6],
    )
    table |> CSV.write("data/deviations_original_no_blocks.table", delim = ' ')
end

function main_original()
    Ns = 20:2:50
    deviations = Array{Float64}(undef, length(Ns), 6)
    bond_dims = similar(deviations, Int)
    for (i, N) in enumerate(Ns)
        Bₙs = [groundstate_Bₙ(N), -N/2 + 2 : 2 : +N/2 - 2]
        truncations = [(; maxdim = 10N), (; maxdim = 20N), (; maxrelerror = 1e-9)]
        for (configuration, (truncation, Bₙ)) in enumerate(Iterators.product(truncations, Bₙs))
            deviation, MPS = study_betheMPS(N, Bₙ, truncation, ())
            bond_dim = maximum(bond_dimension(MPS, bond, Outgoing(:a)) for bond = 1:N-1)
            @show (N, deviation, bond_dim)
            deviations[i, configuration] = deviation
            bond_dims[i, configuration] = bond_dim
        end
    end
    table = Table(
        N = Ns,
        deviation_ground_state_maxdim_10N = deviations[:, 1],
        deviation_ground_state_maxdim_20N = deviations[:, 2],
        deviation_ground_state_truncation_error_e9 = deviations[:, 3],
        deviation_excited_state_maxdim_10N = deviations[:, 4],
        deviation_excited_state_maxdim_20N = deviations[:, 5],
        deviation_excited_state_truncation_error_e9 = deviations[:, 6],
        bond_dim_ground_state_maxdim_10N = bond_dims[:, 1],
        bond_dim_ground_state_maxdim_20N = bond_dims[:, 2],
        bond_dim_ground_state_truncation_error_e9 = bond_dims[:, 3],
        bond_dim_excited_state_maxdim_10N = bond_dims[:, 4],
        bond_dim_excited_state_maxdim_20N = bond_dims[:, 5],
        bond_dim_excited_state_truncation_error_e9 = bond_dims[:, 6],
    )
    table |> CSV.write("data/deviations_original.table", delim = ' ')
end

function main_entanglement_original_groundstate()
    N = 40
    Bₙ = groundstate_Bₙ(N)
    dev, MPS, entanglement = study_betheMPS(N, Bₙ, (; maxdim = 20N), (), bond = N÷2)
    table = Table(entanglement_groundstate = entanglement)
    table |> CSV.write("data/entanglement_original_groundstate_$N.table", delim = ' ')
end

function main_sym()
    Ns = 20:2:40
    deviations = Array{Float64}(undef, length(Ns), 4)
    bond_dims = similar(deviations, Int)
    for (i, N) in enumerate(Ns)
        Bₙs = [groundstate_Bₙ(N), -N/2 + 2 : 2 : +N/2 - 2]
        truncations = [(; maxdim = 20N), (; maxrelerror = 1e-9)]
        for (configuration, (truncation, Bₙ)) in enumerate(Iterators.product(truncations, Bₙs))
            deviation, MPS = study_betheMPS(N, Bₙ, truncation, (), real_symmetric = true)
            bond_dim = maximum(bond_dimension(MPS, bond, Outgoing(:a)) for bond = 1:N-1)
            @show (N, deviation, bond_dim)
            deviations[i, configuration] = deviation
            bond_dims[i, configuration] = bond_dim
        end
    end
    table = Table(
        N = Ns,
        deviation_ground_state_maxdim_20N = deviations[:, 1],
        deviation_ground_state_truncation_error_e9 = deviations[:, 2],
        deviation_excited_state_maxdim_20N = deviations[:, 3],
        deviation_excited_state_truncation_error_e9 = deviations[:, 4],
        bond_dim_ground_state_maxdim_20N = bond_dims[:, 1],
        bond_dim_ground_state_truncation_error_e9 = bond_dims[:, 2],
        bond_dim_excited_state_maxdim_20N = bond_dims[:, 3],
        bond_dim_excited_state_truncation_error_e9 = bond_dims[:, 4],
    )
    table |> CSV.write("data/deviations_sym.table", delim = ' ')
end

function main_optimization()
    Ns = 20:2:30
    deviations = Array{Float64}(undef, length(Ns), 2)
    bond_dims = similar(deviations, Int)
    for (i, N) in enumerate(Ns)
        Bₙs = [groundstate_Bₙ(N), -N/2 + 2 : 2 : +N/2 - 2]
        for (configuration, Bₙ) in enumerate(Bₙs)
            orginal_deviation, deviation, MPS = study_betheMPS(
                N, Bₙ, (; maxdim = 20N), (; maxdim = 40N), optimize = true
            )
            bond_dim = maximum(bond_dimension(MPS, bond, Outgoing(:a)) for bond = 1:N-1)
            @show (N, deviation, bond_dim)
            deviations[i, configuration] = deviation
            bond_dims[i, configuration] = bond_dim
        end
    end
    table = Table(
        N = Ns,
        deviation_ground_state_maxdim_20N_40N = deviations[:, 1],
        deviation_excited_state_maxdim_20N_40N = deviations[:, 2],
        bond_dim_ground_state_maxdim_20N_40N = bond_dims[:, 1],
        bond_dim_excited_state_maxdim_20N_40N = bond_dims[:, 2],
    )
    table |> CSV.write("data/deviations_optimization.table", delim = ' ')
end

function main_optimization_small_bond_dimensions()
    Ns = 20:2:30
    deviations = Array{Float64}(undef, length(Ns), 2)
    bond_dims = similar(deviations, Int)
    for (i, N) in enumerate(Ns)
        Bₙs = [groundstate_Bₙ(N), -N/2 + 2 : 2 : +N/2 - 2]
        for (configuration, Bₙ) in enumerate(Bₙs)
            orginal_deviation, deviation, MPS = study_betheMPS(
                N, Bₙ, (; maxdim = 10N), (; maxdim = 20N), optimize = true
            )
            bond_dim = maximum(bond_dimension(MPS, bond, Outgoing(:a)) for bond = 1:N-1)
            @show (N, deviation, bond_dim)
            deviations[i, configuration] = deviation
            bond_dims[i, configuration] = bond_dim
        end
    end
    table = Table(
        N = Ns,
        deviation_ground_state_maxdim_10N_20N = deviations[:, 1],
        deviation_excited_state_maxdim_10N_20N = deviations[:, 2],
        bond_dim_ground_state_maxdim_10N_20N = bond_dims[:, 1],
        bond_dim_excited_state_maxdim_10N_20N = bond_dims[:, 2],
    )
    table |> CSV.write("data/deviations_optimization_small_bond_dimensions.table", delim = ' ')
end

end
