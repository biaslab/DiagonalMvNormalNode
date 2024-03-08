using RxInfer

struct ScaledChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    τ::T
end

@rule MvNormalDiagonalPrecision{N}((:p, k), Marginalisation) (q_out::PointMass, q_m::PointMass, ) where {N} = begin 
    wishart = @call_rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out=q_out, q_μ=q_m)
    Λ = wishart.invS
    λ = diag(Λ)[k]
    return convert(GammaShapeRate, ScaledChisq(N + N - 1.0, λ / (N - N + 1)))
end

