using RxInfer
using LinearAlgebra
using BenchmarkTools
using SpecialFunctions


include("node.jl")

@model function normal_with_precisions(d, n)
    y = datavar(Vector{Float64}, n)    
    α = randomvar(d)
    for i in 1:d
        α[i] ~ Gamma(1, 1)
    end
    alpha = Tuple([α[j] for j in 1:d])
    for i in 1:n
        y[i] ~ MvNormalDiagonalPrecision(zeros(d), alpha) 
    end
end

d = 5
n = 100
prec = diagm(rand(d)) 
data = [rand(MvNormalMeanPrecision(zeros(d), prec)) for _ in 1:n]


result = infer(model=normal_with_precisions(d, n), data=(y=data,))

struct ScaledChisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    τ::T
end

Distributions.pdf(d::ScaledChisq, x::Real) = Distributions.pdf(Chisq(d.ν), x * d.ν * d.τ) * d.ν * d.τ
Distributions.logpdf(d::ScaledChisq, x::Real) = Distributions.logpdf(Chisq(d.ν), x * d.ν * d.τ) + log(d.ν * d.τ)

Base.convert(::Type{GammaShapeRate}, x::ScaledChisq) = GammaShapeRate(x.ν/2, ((x.τ * x.ν) / 2))
Base.convert(::Type{GammaShapeScale}, x::ScaledChisq) = GammaShapeScale(x.ν/2, inv((x.τ * x.ν) / 2))

@rule MvNormalDiagonalPrecision{N}((:p, k), Marginalisation) (q_out::PointMass, q_m::PointMass, ) where {N} = begin 
    wishart = @call_rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out=q_out, q_μ=q_m)
    Λ = diag(wishart.invS)
    λ = Λ[k]
    ν = 2 * N - 1.0
    τ = λ 
    ν, τ = promote(ν, τ)
    return ScaledChisq(ν, τ)
end

BayesBase.prod(::GenericProd, left::GammaDistributionsFamily, right::ScaledChisq) = begin
    n = 1000
    samples = rand(left, n)
    weights = pdf.(right, samples)
    Distributions.fit(Gamma, samples, weights)
end
