using CUDA
using RKIntegrators
using Test

CUDA.allowscalar(false)


# https://discourse.julialang.org/t/using-allocated-to-track-memory-allocations/4617/8
allocated(func, integ, u, t, dt) = @allocated func(integ, u, t, dt)


algs = [RK3(), SSPRK3(), RK4(), Tsit5(), RK45(), DP5(), ATsit5()]

begin
    @testset "Out-of-place" begin include("oop.jl") end
    @testset "In-place" begin include("iip.jl") end
    @testset "In-place CUDA" begin include("iip_cuda.jl") end
end
