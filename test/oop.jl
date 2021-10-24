function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    u[1] = integ.u0
    for i=1:Nt-1
        u[i+1] = rkstep(integ, u[i], t[i], dt)
    end
    return nothing
end


Nt = 100
t = range(0.0, 3.0, length=Nt)

a = 2.0

u0s = [10.0, 10.0 + 10.0im]

for u0 in u0s
    uth = @. u0 * exp(-a * t)

    p = (a,)
    prob = Problem(func, u0, p)

    u = zeros(eltype(u0), Nt)
    for alg in algs
        integ = Integrator(prob, alg)
        solve!(u, t, integ)

        @test isapprox(u, uth, rtol=1e-5)
        @test allocated(solve!, u, t, integ) == 0
    end
end
