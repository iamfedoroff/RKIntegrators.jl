function func(du, u, p, t)
    a, = p
    for i=1:length(u)
        du[i] = -a * u[i]
    end
    return nothing
end


function solve!(u, t, integs)
    Nu, Np, Nt = size(u)
    nth = min(Np, 256)
    nbl = cld(Np, nth)
    @cuda threads=nth blocks=nbl solve_kernel(u, t, integs)
    return nothing
end


function solve_kernel(u, t, integs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Np = length(integs)
    Nt = length(t)
    dt = t[2] - t[1]

    for ip=id:stride:Np
        integ = integs[ip].integ
        utmp = integs[ip].utmp

        ipre = CartesianIndices(size(utmp))

        for iu in ipre
            utmp[iu] = integ.u0[iu]
            u[iu, ip, 1] = utmp[iu]
        end

        for it=1:Nt-1
            rkstep!(integ, utmp, t[it], dt)

            for iu in ipre
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
u0s = [
    10 * CUDA.ones(Nu),
    10 * CUDA.ones(ComplexF32, Nu),
    10 * CUDA.ones((Nu, Nu)),
]

for u0 in u0s
    s = size(u0)
    ipre = CartesianIndices(s)

    uth = zeros(eltype(u0), (s..., Np, Nt))
    u0cpu = collect(u0)
    for j=1:Nt
    for i=1:Np
        @. uth[ipre,i,j] = u0cpu[ipre] * exp(-a[i] * t[j])
    end
    end

    probs = Array{Problem}(undef, Np)
    for i=1:Np
        p = (a[i], )
        probs[i] = Problem(func, u0, p)
    end

    u = CUDA.zeros(eltype(u0), (s..., Np, Nt))
    for alg in algs
        integs = integrator_ensemble(probs, alg)
        cuintegs = CuArray(cudaconvert.(integs))
        GC.@preserve integs begin
            solve!(u, t, cuintegs)
        end

        @test isapprox(collect(u), uth, rtol=1e-4)
    end
end
