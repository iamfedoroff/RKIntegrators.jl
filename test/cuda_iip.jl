function func(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    ipre = CartesianIndices(size(utmp))
    @. utmp = integ.u0
    @. u[ipre, 1] = utmp
    for i=1:Nt-1
        rkstep!(integ, utmp, t[i], dt)
        @. u[ipre, i+1] = utmp
    end
    return nothing
end


Nt = 100
t = range(0f0, 3f0, length=Nt)

a = 2f0

Nu = 100
u0s = [
    10 * CUDA.ones(Nu),
    10 * CUDA.ones(ComplexF32, Nu),
    10 * CUDA.ones(Nu, Nu),
]

for u0 in u0s
    s = size(u0)
    ipre = CartesianIndices(s)

    uth = zeros(eltype(u0), (s..., Nt))
    u0cpu = collect(u0)
    for i=1:Nt
        @. uth[ipre, i] = u0cpu[ipre] * exp(-a * t[i])
    end

    p = (a,)
    prob = Problem(func, u0, p)

    u = CUDA.zeros(eltype(u0), (s..., Nt))
    utmp = CUDA.zeros(eltype(u0), s)
    for alg in algs
        integ = Integrator(prob, alg)
        solve!(u, utmp, t, integ)

        @test isapprox(collect(u), uth, rtol=1e-5)
        @test allocated(solve!, u, utmp, t, integ) == 0
    end
end
