function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integ)
    Nt = length(t)
    dt = t[2] - t[1]
    u[1] = integ.prob.u0
    for i=1:Nt-1
        u[i+1] = rkstep(integ, u[i], t[i], dt)
    end
    return nothing
end


t = range(0.0, 3.0, length=100)
Nt = length(t)
dt = t[2] - t[1]

a = 2.0
p = (a,)

u0s = [10.0, 10.0 + 10.0im]

for u0 in u0s
    prob = Problem(func, u0, p)

    uth = @. u0 * exp(-a * t)

    u = zeros(eltype(u0), Nt)

    for alg in algs
        integ = Integrator(prob, alg)
        solve!(u, t, integ)

        @test isapprox(u, uth, rtol=1e-5)

        @test allocated(rkstep, integ, u0, t[1], dt) == 0
    end
end
