module BetheNetwork

export B_MPO, symMPO
export bethe_network, betheMPS, optimize_betheMPS!

using BlockTensors
using BlockTensors: prune!
using BlockTensors.TensorChain
using BlockTensors.MatrixProductStates
using ..Pauli
using ..LaxOperators


function B_MPO(N::Integer, lax::Tensor, auxiliary, types...)
    MPO = createchain(lax, N) do n, connector
        matching(auxiliary, connector) ? () : (; n)
    end
    connectchain!(MPO, auxiliary)
    T, S = eltype(MPO).parameters
    MPO = convert(Vector{Tensor{T,S}}, MPO)
    a_in = only(matching(Incoming(auxiliary), MPO[begin]))
    MPO[begin] = MPO[begin] * spindown(dual(a_in, connect = true), types...)
    a_out = only(matching(Outgoing(auxiliary), MPO[end]))
    MPO[end] = MPO[end] * spinup(dual(a_out, connect = true), types...)
    return MPO
end
function B_MPO(N::Integer, u::Number, physical, auxiliary, types...)
    B_MPO(N, lax_operator(u, physical, auxiliary, types...), auxiliary, types...)
end

function symMPO(N::Integer, u::Number, physical, auxiliary, types...)
    p = Space(physical)
    a = Space(auxiliary, u = round(abs(u), digits = 3))
    a1 = Space(a, i = 1)
    a2 = Space(a, i = 2)
    L⁻1 = lax_operator(-u, Outgoing(p), Incoming(p), Outgoing(a1), Incoming(a1), types...)
    L⁻2 = lax_operator(-u, Outgoing(p), Incoming(p), Outgoing(a2), Incoming(a2), types...)
    L⁺1 = lax_operator(+u, Outgoing(p), Incoming(p), Outgoing(a1), Incoming(a1), types...)
    L⁺2 = lax_operator(+u, Outgoing(p), Incoming(p), Outgoing(a2), Incoming(a2), types...)
    connect!(L⁻1, L⁺2, physical)
    connect!(L⁺1, L⁻2, physical)
    T, S = typeof(L⁺1).parameters
    A1 = Tensor(Dict((zero(S), zero(S)) => T[1 -im; +im 1] / 2), Outgoing(a), Incoming(a))
    A2 = Tensor(Dict((zero(S), zero(S)) => T[1 +im; -im 1] / 2), Outgoing(a), Incoming(a))
    as = a1, a2, a
    O1 = mergelegs(L⁻1 * L⁺2 * A1, Outgoing.(as), Incoming.(as))
    O2 = mergelegs(L⁺1 * L⁻2 * A2, Outgoing.(as), Incoming.(as))
    O = convert(Tensor{real(T), S}, O1 + O2)
    prune!(O)
    v = Tensor(Dict(tuple(zero(S)) => real(T)[1, 0]), Outgoing(a))
    auxket = mergelegs(
        spindown(Outgoing(a1), real(T), S) * spindown(Outgoing(a2), real(T), S) * v,
        Outgoing.(as)
    )
    auxbra = mergelegs(
        spinup(Outgoing(a1), real(T), S)' * spinup(Outgoing(a2), real(T), S)' * v',
        Incoming.(as)
    )
    MPO = createchain(O, N) do n, connector
        matching(physical, connector) ? (; n) : ()
    end
    connectchain!(MPO, auxiliary)
    MPO = convert(Vector{Tensor{real(T), S}}, MPO)
    connect!(auxket, MPO[begin], auxiliary)
    MPO[begin] *= auxket
    connect!(MPO[end], auxbra, auxiliary)
    MPO[end] *= auxbra
    return MPO
end

function bethe_network(N::Integer, spectral_parameters, physical, auxiliary, types...)
    operators = lax_operators(spectral_parameters, physical, auxiliary, types...)
    bethe_network(B_MPO.(N, operators, auxiliary, types), physical)
end
function bethe_network(MPOs, physical)
    network = cat(reshape.(MPOs, 1, :)..., dims = 1)
    T, S = eltype(network).body.parameters
    connectchain!(network, physical, dim = 1)
    for n in axes(network, 2)
        p = only(matching(Incoming(physical), network[begin, n]))
        network[begin, n] = network[begin, n] * spinup(dual(p, connect = true), T, S)
    end
    return network
end

function betheMPS(network, auxiliary, bond = nothing; truncation...)
    MPS = network[begin, :]
    canonicalize!(MPS, eachindex(MPS), Outgoing(auxiliary))
    if bond ≢ nothing
        entanglements = [entanglement_entropy(MPS, bond, Outgoing(auxiliary))]
        sizehint!(entanglements, size(network, 1))
    else
        entanglements = nothing
    end
    for k in axes(network, 1)[begin + 1 : end]
        MPS = contractchains(MPS, view(network, k, :), auxiliary)
        canonicalize!(MPS, eachindex(MPS), Outgoing(auxiliary))
        if !isempty(truncation)
            canonicalize!(MPS, reverse(eachindex(MPS)), Incoming(auxiliary); truncation...)
        end
        if bond ≢ nothing
            push!(entanglements, entanglement_entropy(MPS, bond, Outgoing(auxiliary)))
        end
    end
    return MPS, entanglements
end

function betheMPS_projection(network, physical, auxiliary; truncation...)
    MPS = similar(network, size(network, 2))
    for k in axes(network, 1)
        canonicalize!(view(network, k, :), reverse(axes(network, 2)), Incoming(auxiliary))
    end
    contraction = network[:, begin]
    projection_ket = deepcopy(contraction)
    for tensor in projection_ket, leg in matching(Space(physical), tensor.legs)
        addtags!(leg, projection = true)
    end
    projection_bra = adjoint.(projection_ket)
    MPS[begin], projection_bra, projection_ket = canonicalize_project(
        contraction, projection_bra, projection_ket, physical, truncation
    )
    for n in axes(network, 2)[begin + 1 : end - 1]
        T = mergelegs(
            projection_ket[end] * network[end, n], 
            union(
                matching(Incoming(physical), projection_ket[end]), 
                matching(Incoming(physical), network[end, n])
            )
        )
        projection_ket = contractchains(projection_ket, view(network, :, n), physical)
        projection_bra = contractchains(projection_bra, adjoint.(view(network, :, n)), physical)
        contraction .= copy.(projection_ket)
        contraction[end] = T
        MPS[n], projection_bra, projection_ket = canonicalize_project(
            contraction, projection_bra, projection_ket, physical, truncation
        )
    end
    T = 1
    for (Tket, L) in zip(projection_ket, view(network, :, lastindex(network, 2)))
        T = (T * L) * Tket
    end
    MPS[end] = T
    for n in eachindex(MPS)
        new_legs = map(MPS[n].legs) do leg
            if (:projection => true) in leg.connector.space.tags
                Leg(typeof(leg.connector)(auxiliary), leg.dimensions)
            else
                leg
            end
        end
        MPS[n] = Tensor(MPS[n].components, new_legs)
    end
    connectchain!(reverse(MPS), auxiliary)
    return MPS
end

function canonicalize_project(contraction, projection_bra, projection_ket, physical, truncation)
    canonicalize!(
        projection_ket, eachindex(projection_ket), Outgoing(physical); 
        normalize = false
    )
    exchangegauge(projection_ket[end], projection_bra[end], Outgoing(physical))
    canonicalize!(
        projection_bra, reverse(eachindex(projection_bra)), Outgoing(physical);
        normalize = false
    )
    canonicalize!(
        projection_bra, eachindex(projection_bra), Incoming(physical);
        normalize = false, truncation...
    )
    exchangegauge(
        projection_bra[end], projection_ket[end], Incoming(physical); truncation...
    )
    canonicalize!(
        projection_ket, reverse(eachindex(projection_ket)), Incoming(physical); 
        normalize = false, truncation...
    )
    canonicalize!(
        projection_ket, eachindex(projection_ket), Outgoing(physical); 
        normalize = false
    )
    T = 1
    for (Tket, Tbra) in zip(contraction, projection_bra)
        T = (T * Tbra) * Tket
    end
    exchangegauge(projection_ket[end], projection_bra[end], Outgoing(physical))
    return T, projection_bra, projection_ket
end

function optimize_betheMPS!(MPS, network, physical, auxiliary; truncation...)
    @assert size(network, 2) == length(MPS)
    scaling = ones(float(real(eltype(eltype(MPS)))), size(network, 2))
    contractions = similar(network, size(network, 1) + 1, size(network, 2))
    N = axes(network, 2)
    Nbegin = N[begin : end - 2]
    Nmiddle = N[begin + 1 : end - 1]
    Nend = N[begin + 2 : end]
    MPS[Nbegin[begin]], MPS[Nmiddle[begin]] = exchangegauge(
        MPS[Nbegin[begin]], MPS[Nmiddle[begin]], Outgoing(auxiliary)
    )
    contractions[axes(network, 1), [begin, end]] .= network[:, [begin, end]]
    contractions[end, begin] = MPS[begin]'
    for ns in zip(Nbegin, Nmiddle, Nend)
        updatelocally!(
            MPS, contractions, scaling, network, 
            physical, Outgoing(auxiliary), ns...; 
            truncation...
        ) do MPS, contractions, auxiliary, nprev, n, nnext
            MPS[n], MPS[nnext] = exchangegauge(MPS[n], MPS[nnext], auxiliary)
            nothing
        end
    end
    norm = updatelocally!(
        MPS, contractions, Incoming(auxiliary), Nmiddle[end], Nend[end]
    )
    norm *= scaling[Nmiddle[end]]
    @show norm; norms = [norm]
    sizehint!(norms, 2 * length(MPS))
    while length(norms) < length(MPS) || !isapprox(norms[end], norms[end - length(MPS) + 1])
        for (nprev, n, nnext) in zip(reverse(Nend), reverse(Nmiddle), reverse(Nbegin)) 
            norm = updatelocally!(
                optimize!, MPS, contractions, scaling, network, 
                physical, Incoming(auxiliary), nprev, n, nnext;
                truncation...
            )
            norm *= scaling[nprev] * scaling[nnext]
            @show norm; push!(norms, norm)
        end
        norm = updatelocally!(
            MPS, contractions, Outgoing(auxiliary), Nmiddle[begin], Nbegin[begin]
        )
        norm *= scaling[Nmiddle[begin]]
        @show norm; push!(norms, norm)
        for (nprev, n, nnext) in zip(Nbegin, Nmiddle, Nend)
            norm = updatelocally!(
                optimize!, MPS, contractions, scaling, network, 
                physical, Outgoing(auxiliary), nprev, n, nnext;
                truncation...
            )
            norm *= scaling[nprev] * scaling[nnext]
            @show norm; push!(norms, norm)
        end
        norm = updatelocally!(
            MPS, contractions, Incoming(auxiliary), Nmiddle[end], Nend[end]
        )
        norm *= scaling[Nmiddle[end]]
        @show norm; push!(norms, norm)
        if length(norms) > 9 * length(MPS) break end
    end
    return norms, contractions
end

function optimize!(MPS, contractions, auxiliary, nprev, n, nnext)
    contraction = 1
    for k in axes(contractions, 1)[begin : end - 1]
        contraction = (contractions[k, n] * contraction) * contractions[k, nnext]
    end
    MPS[n], R = qr(
        (contractions[end, nprev] * contraction) * contractions[end, nnext], 
        auxiliary
    )
    norm = √real(R'R)
    MPS[nnext] = R / norm * MPS[nnext]
    return norm
end

function updatelocally!(
    update!, MPS, contractions, scaling, network, 
    physical, auxiliary, nprev, n, nnext; truncation...
)
    Knet = axes(network, 1)
    contractions[Knet, n] .= view(contractions, Knet, nprev) .* view(network, :, n)
    result = update!(MPS, contractions, auxiliary, nprev, n, nnext)
    contractions[end, n] = contractions[end, nprev] * MPS[n]'
    K = axes(contractions, 1)
    for k in K 
        contractions[k, n] = mergelegs(
            contractions[k, n],
            union(
                matching(Outgoing(physical), contractions[k, nprev]),
                matching(Outgoing(physical), contractions[k, n])
            ),
            union(
                matching(Incoming(physical), contractions[k, nprev]),
                matching(Incoming(physical), contractions[k, n])
            )
        )
    end
    canonicalize!(
        view(contractions, :, n), reverse(K), Incoming(physical), normalize = false
    )
    norm = canonicalize!(
        view(contractions, :, n), K, Outgoing(physical), normalize = true; truncation...
    )
    scaling[n] = scaling[nprev] * norm
    return result
end

function updatelocally!(
    MPS, contractions, auxiliary, nprev, n
)
    contraction = 1
    for k in axes(contractions, 1)[begin : end - 1]
        contraction = contractions[k, nprev] * (contraction * contractions[k, n])
    end
    MPS[n], R = qr(contractions[end, nprev] * contraction, auxiliary)
    norm = √real(R'R)
    MPS[nprev] = MPS[nprev] * R / norm
    contractions[end, n] = MPS[n]'
    return norm
end


end
