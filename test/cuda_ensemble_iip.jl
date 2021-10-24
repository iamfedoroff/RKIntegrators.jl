function func(du, u, p, t)
    a, = p
    for i=1:length(u)
        du[i] = -a * u[i]
    end
    return nothing
end


function solve!(u, t, integs)
    Nu, Np, Nt = size(u)
    nthreads = min(Np, 256)
    nblocks = cld(Np, nthreads)

    GC.@preserve integs begin
        cuintegs = CuArray(cudaconvert.(integs))
        @cuda threads=nthreads blocks=nblocks solve_kernel(u, t, cuintegs)
    end
    return nothing
end


function solve_kernel(u, t, integs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Nu, Np, Nt = size(u)
    dt = t[2] - t[1]

    for ip=id:stride:Np
        integ = integs[ip].integ
        utmp = integs[ip].utmp

        for iu=1:Nu
            utmp[iu] = integ.u0[iu]
            u[iu, ip, 1] = utmp[iu]
        end

        for it=1:Nt-1
            rkstep!(integ, utmp, t[it], dt)

            for iu=1:Nu
                u[iu,ip,it+1] = utmp[iu]
            end
        end
    end
    return nothing
end


Nt = 100
t = range(0f0, 3f0, length=Nt)

Np = 1000
a = range(1f0, 3f0, length=Np)

Nu = 2
u0s = [10 * CUDA.ones(Float32, Nu), 10 * CUDA.ones(ComplexF32, Nu)]

for u0 in u0s
    uth = zeros(eltype(u0), (Nu, Np, Nt))
    u0cpu = collect(u0)
    for ip=1:Np
    for iu=1:Nu
        @. uth[iu,ip,:] = u0cpu[iu] * exp(-a[ip] * t)
    end
    end

    probs = Array{Problem}(undef, Np)
    for i=1:Np
        p = (a[i], )
        probs[i] = Problem(func, u0, p)
    end

    u = CUDA.zeros(eltype(u0), (Nu, Np, Nt))
    for alg in algs
        integs = integrator_ensemble(probs, alg)
        solve!(u, t, integs)

        @test isapprox(collect(u), uth, rtol=1e-4)
    end
end
