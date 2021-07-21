using CUDA
using RKIntegrators
using Test

# https://discourse.julialang.org/t/using-allocated-to-track-memory-allocations/4617/8
allocated(func, integ, u, t, dt) = @allocated func(integ, u, t, dt)
cuallocated(func, integ, u, t, dt) = CUDA.@allocated func(integ, u, t, dt)


algs = [RK3(), SSPRK3(), RK4(), Tsit5(), RK45(), DP5(), ATsit5()]

@testset "CPU" begin
    include("oop.jl")
    include("iip.jl")
end

@testset "GPU" begin
    if has_cuda()
        CUDA.allowscalar(false)
        include("iip_cuda.jl")
    end
end
