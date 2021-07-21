function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, kintegs)
    Nr, Nt = size(u)
    nthreads = min(Nr, 256)
    nblocks = cld(Nr, nthreads)
    @cuda threads=nthreads blocks=nblocks solve_kernel(u, t, kintegs)
end


function solve_kernel(u, t, kintegs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Nr, Nt = size(u)
    dt = t[2] - t[1]

    for ir=id:stride:Nr
        integ = kintegs[ir].integ

        u[ir, 1] = integ.prob.u0
        for it=1:Nt-1
            u[ir,it+1] = rkstep(integ, u[ir,it], t[it], dt)
        end
    end
    return nothing
end


t = range(0f0, 3f0, length=100)
Nt = length(t)
dt = t[2] - t[1]

Nr = 3
a = range(1f0, 3f0, length=Nr)

u0 = 10f0

probs = Array{Problem}(undef, Nr)
for ir=1:Nr
    local p = (a[ir], )
    probs[ir] = Problem(func, u0, p)
end

uth = zeros(Float32, (Nr, Nt))
for ir=1:Nr
    @. uth[ir, :] = u0 * exp(-a[ir] * t)
end

u = CUDA.zeros(Float32, (Nr, Nt))

for alg in algs
    kintegs = integrator_ensemble(probs, alg)
    solve!(u, t, kintegs)

    @test isapprox(collect(u), uth, rtol=1e-4)
    @test solve_allocated(u, t, kintegs) == 0
end
