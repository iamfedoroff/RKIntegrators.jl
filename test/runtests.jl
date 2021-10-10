using CUDA
using RKIntegrators
using Test

# https://discourse.julialang.org/t/using-allocated-to-track-memory-allocations/4617/8
# CUDA version of @allocated allows to solve this issue
allocated(func, args...) = CUDA.@allocated func(args...)


algs = [RK3(), SSPRK3(), RK4(), Tsit5(), RK45(), DP5(), ATsit5()]

@testset "CPU" begin
    include("oop.jl")
    include("iip.jl")
    include("ensemble_oop.jl")
    include("ensemble_iip.jl")
end

@testset "CUDA" begin
    if has_cuda()
        CUDA.allowscalar(false)
        include("cuda_iip.jl")
        # include("oop_cuda_kernel.jl")
        # include("iip_cuda_kernel.jl")
    end
end
