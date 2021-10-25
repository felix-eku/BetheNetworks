module LaxOperators

using BlockTensors
using BlockTensors.TensorChain
using ..Pauli

export lax_operator, lax_operators

function lax_operator(u, physical, auxiliary, types...)
    scaling = u * pauli(0, physical, types...) * pauli(0, auxiliary, types...)
    return scaling + (im/2) * sum(1:3) do index
        pauli(index, physical, types...) * pauli(index, auxiliary, types...)
    end
end

function lax_operators(spectral_parameters, physical, auxiliary, types...)
    operators = [
        (
            p = copy(physical); a = copy(auxiliary); 
            addtags!(a; u = round(u, digits = 3)); 
            lax_operator(u, p, a, types...)
        )
        for u in spectral_parameters
    ]
    K = LinearIndices(operators)
    for (k1, k2) in zip(K[begin : end - 1], K[begin + 1 : end])
        connect!(
            only(matching(Outgoing(physical), operators[k1])),
            only(matching(Incoming(physical), operators[k2])),
        )
    end
    return operators
end

end
