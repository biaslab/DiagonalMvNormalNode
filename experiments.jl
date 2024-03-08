using RxInfer
using LinearAlgebra
using BenchmarkTools

include("scaled_chisq.jl")
include("node.jl")
include("rules.jl")

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