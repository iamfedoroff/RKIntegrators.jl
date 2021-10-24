function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integs)
    Np  = length(integs)
    nth = min(Np, 256)
    nbl = cld(Np, nth)
    @cuda threads=nth blocks=nbl solve_kernel(u, t, integs)
end


function solve_kernel(u, t, integs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Np, Nt = size(u)
    dt = t[2] - t[1]

    for ip=id:stride:Np
        integ = integs[ip].integ

        u[ip,1] = integ.u0
        for it=1:Nt-1
            u[ip,it+1] = rkstep(integ, u[ip,it], t[it], dt)
        end
    end
    return nothing
end


Nt = 100
t = range(0f0, 3f0, length=Nt)

Np = 1000
a = range(1f0, 3f0, length=Np)

u0s = [10f0, 10f0 + 10f0im]

for u0 in u0s
    uth = zeros(eltype(u0), (Np, Nt))
    for i=1:Np
        @. uth[i,:] = u0 * exp(-a[i] * t)
    end

    probs = Array{Problem}(undef, Np)
    for i=1:Np
        p = (a[i], )
        probs[i] = Problem(func, u0, p)
    end

    u = CUDA.zeros(eltype(u0), (Np, Nt))
    for alg in algs
        integs = integrator_ensemble(probs, alg)
        cuintegs = CuArray(cudaconvert.(integs))
        GC.@preserve integs begin
            solve!(u, t, cuintegs)
        end

        @test isapprox(collect(u), uth, rtol=1e-4)
    end
end
