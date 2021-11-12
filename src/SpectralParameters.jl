module SpectralParameters

export solveBAE, groundstate_Bₙ

"""Calculate (real) solutions of the Bethe Ansatz equations (BAE).

Solve the Bethe Ansatz equations

    ((uₙ+i/2) / (uₙ-i/2))^N = Π_{j=1≠n}^M (uₙ-uⱼ+i) / (uₙ-uⱼ-i)

for a system of size `N` (= number of spins) and M magnons
by iterating the logarithmic equation

    uₙ = tan(π/2N Bₙ + 1/N Σ_{j=1≠n}^M arctan(uₙ-uⱼ)) / 2

with Bethe quantum numbers `Bₙ`_{n=1}^M.

Adapted from "Karbach, Hu, Mueller: Introduction to the Bethe Ansatz II (2008)"
equation (9), with uₙ = zₙ/2 and Bₙ = 2Iₙ.
"""
function solveBAE(N::Integer, Bₙ::AbstractVector{<:Real}; tolerance...)
    phases =  pi * Bₙ / 2N
    u_old = zeros(axes(phases))
    u = tan.(phases) / 2
    while !all( isapprox(uₙ, u_oldₙ; rtol=eps(), tolerance...) 
                for (uₙ, u_oldₙ) in zip(u, u_old) )
        u_old .= u
        u .= tan.(phases .+ scattering.(u, Ref(u))./N) ./ 2
    end
    return u
end
scattering(uₙ, u) = sum(atan(uₙ - uⱼ) for uⱼ in u)


"""
Configuration of Bₙ for the antiferromagnetic ground state with `N` spins.
"""
function groundstate_Bₙ(N::Integer)
    iseven(N) || throw(ArgumentError("N is not even"))
    return (-N/2 + 1) : 2 : (N/2 - 1)
end

end
