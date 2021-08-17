module LaxOperators

using BlockTensors
using BlockTensors.TensorChain
using ..Pauli

export lax_operator, lax_operators

function lax_operator(spectral_parameter, physical, auxiliary)
    scaling = spectral_parameter * pauli(0, physical) * pauli(0, auxiliary)
    return scaling + (im/2) * sum(1:3) do index
        pauli(index, physical) * pauli(index, auxiliary)
    end
end

function lax_operators(spectral_parameters, physical, auxiliary)
    operators = [
        (
            p = copy(physical); a = copy(auxiliary); 
            addtags!(a; u = round(u, digits = 3)); 
            lax_operator(u, p, a)
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
