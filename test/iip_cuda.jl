function func(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, utmp, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    @. utmp = integ.prob.u0
    @. u[:, 1] = utmp
    for i=1:Nt-1
        rkstep!(integ, utmp, t[i], dt)
        @. u[:, i+1] = utmp
    end
    return nothing
end


t = range(0f0, 3f0, length=100)
Nt = length(t)
dt = t[2] - t[1]

a = 2f0
p = (a,)

Nu = 1000
u0s = [10 * CUDA.ones(Float32, Nu), 10 * CUDA.ones(ComplexF32, Nu)]

for u0 in u0s
    prob = Problem(func, u0, p)

    uth = zeros(eltype(u0), (Nu, Nt))
    u0cpu = collect(u0)
    for i=1:Nu
        @. uth[i, :] = u0cpu[i] * exp(-a * t)
    end

    u = CUDA.zeros(eltype(u0), (Nu, Nt))
    utmp = CUDA.zeros(eltype(u0), Nu)

    for alg in algs
        integ = Integrator(prob, alg)
        solve!(u, utmp, t, integ)

        @test isapprox(collect(u), uth, rtol=1e-5)

        @test cuallocated(rkstep!, integ, utmp, t[1], dt) == 0
    end
end
