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


t = range(0.0, 3.0, length=100)
Nt = length(t)
dt = t[2] - t[1]

Nu = 2
u0 = 10.0 * ones(Nu)
a = 2.0
p = (a,)
prob = Problem(func, u0, p)

uth = zeros((Nu, Nt))
for i=1:Nu
    @. uth[i, :] = u0[i] * exp(-a * t)
end

u = zeros((Nu, Nt))
utmp = zeros(Nu)

for alg in algs
    integ = Integrator(prob, alg)
    solve!(u, utmp, t, integ)

    @test isapprox(u, uth, rtol=1e-5)

    @test allocated(rkstep!, integ, utmp, t[1], dt) == 0
end
