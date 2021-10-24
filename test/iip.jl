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
t = range(0.0, 3.0, length=Nt)

a = 2.0

Nu = 2
u0s = [10 * ones(Nu), 10 * ones(ComplexF64, Nu), 10 * ones((Nu, Nu))]

for u0 in u0s
    s = size(u0)
    ipre = CartesianIndices(s)

    uth = zeros(eltype(u0), (s..., Nt))
    for i=1:Nt
        @. uth[ipre, i] = u0[ipre] * exp(-a * t[i])
    end

    p = (a,)
    prob = Problem(func, u0, p)

    u = zeros(eltype(u0), (s..., Nt))
    utmp = zeros(eltype(u0), s)
    for alg in algs
        integ = Integrator(prob, alg)
        solve!(u, utmp, t, integ)

        @test isapprox(u, uth, rtol=1e-5)
        @test allocated(solve!, u, utmp, t, integ) == 0
    end
end
