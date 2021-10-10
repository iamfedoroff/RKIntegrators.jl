function func(u, p, t)
    a, = p
    du = -a * u
    return du
end


function solve!(u, t, integs)
    Np, Nt = size(u)
    dt = t[2] - t[1]

    for i=1:Np
        integ = integs[i].integ

        u[i,1] = integ.prob.u0
        for j=1:Nt-1
            u[i,j+1] = rkstep(integ, u[i,j], t[j], dt)
        end
    end
    return nothing
end


Nt = 100
t = range(0.0, 3.0, length=Nt)

Np = 10
a = range(1.0, 3.0, length=Np)

u0s = [10.0, 10.0 + 10.0im]

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

    u = zeros(eltype(u0), (Np, Nt))
    for alg in algs
        integs = integrator_ensemble(probs, alg)
        solve!(u, t, integs)

        @test isapprox(u, uth, rtol=1e-5)
        @test allocated(solve!, u, t, integs) == 0
    end
end
