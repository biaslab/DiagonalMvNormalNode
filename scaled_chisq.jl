using Distributions
using RxInfer

Base.convert(::Type{GammaShapeRate}, x::Chisq) = GammaShapeRate(x.ν/2, 2)
Base.convert(::Type{GammaShapeScale}, x::Chisq) = GammaShapeRate(x.ν/2, 1/2)

struct ScaledChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    τ::T
end
Distributions.pdf(d::ScaledChisq, x::Real) = pdf(Chisq(d.ν), x * (d.τ *d.ν))*(d.ν * d.τ)
Distributions.logpdf(d::ScaledChisq, x::Real) = logpdf(Chisq(d.ν), x * (d.τ *d.ν)) + log(d.ν * d.τ)

Base.convert(::Type{GammaShapeRate}, x::ScaledChisq) = GammaShapeRate(x.ν/2, ((x.τ * x.ν) / 2))
Base.convert(::Type{GammaShapeScale}, x::ScaledChisq) = GammaShapeScale(x.ν/2, inv((x.τ * x.ν) / 2))