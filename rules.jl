using RxInfer

struct ScaledChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    τ::T
end

@rule MvNormalDiagonalPrecision{N}((:p, k), Marginalisation) (q_out::Any, q_m::Any, ) where {N} = begin 
    wishart = @call_rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out=q_out, q_μ=q_m)
    Λ = wishart.invS
    λ = diag(Λ)[k]
    return convert(GammaShapeRate, ScaledChisq(N + N - 1.0, λ / (N - N + 1)))
end

@rule MvNormalDiagonalPrecision{N}(:out, Marginalisation) (q_m::Any, q_p::ManyOf{N, Any}, ) where {N} = begin
    return MvNormalMeanPrecision(mean(q_m), diagm(collect(mean.(q_p))))
end

@rule MvNormalDiagonalPrecision{N}(:m, Marginalisation) (q_out::Any, q_p::ManyOf{N, Any}, ) where {N} = begin
    return MvNormalMeanPrecision(mean(q_out), diagm(collect(mean.(q_p))))
end

