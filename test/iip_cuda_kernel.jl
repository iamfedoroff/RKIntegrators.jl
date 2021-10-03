function func(du, u, p, t)
    a, = p

    Nu = length(u)
    for iu=1:Nu
        du[iu] = -a * u[iu]
    end
    return nothing
end


function solve!(u, t, kintegs)
    Nu, Nr, Nt = size(u)
    nthreads = min(Nr, 256)
    nblocks = cld(Nr, nthreads)
    @cuda threads=nthreads blocks=nblocks solve_kernel(u, t, kintegs)
end


function solve_kernel(u, t, kintegs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Nu, Nr, Nt = size(u)
    dt = t[2] - t[1]

    for ir=id:stride:Nr
        integ = kintegs[ir].integ
        utmp = kintegs[ir].utmp

        for iu=1:Nu
            utmp[iu] = integ.prob.u0[iu]
            u[iu, ir, 1] = utmp[iu]
        end

        for it=1:Nt-1
            rkstep!(integ, utmp, t[it], dt)
            for iu=1:Nu
                u[iu,ir,it+1] = utmp[iu]
            end
        end
    end
    return nothing
end


t = range(0f0, 3f0, length=100)
Nt = length(t)
dt = t[2] - t[1]

Nr = 3
a = range(1f0, 3f0, length=Nr)

Nu = 2
u0s = [10 * CUDA.ones(Float32, Nu), 10 * CUDA.ones(ComplexF32, Nu)]

for u0 in u0s
    probs = Array{Problem}(undef, Nr)
    for ir=1:Nr
        local p = (a[ir], )
        probs[ir] = Problem(func, u0, p)
    end

    uth = zeros(eltype(u0), (Nu, Nr, Nt))
    u0cpu = collect(u0)
    for it=1:Nt
    for ir=1:Nr
    for iu=1:Nu
        uth[iu,ir,it] = u0cpu[iu] * exp(-a[ir] * t[it])
    end
    end
    end

    u = CUDA.zeros(eltype(u0), (Nu, Nr, Nt))

    for alg in algs
        kintegs = integrator_ensemble(probs, alg)
        solve!(u, t, kintegs)

        @test isapprox(collect(u), uth, rtol=1e-4)
        # @test solve_allocated(u, t, kintegs) == 0   # sometimes leads to "error in running finalizer"
    end
end
