function func(du, u, p, t)
    a, = p
    @. du = -a * u
    return nothing
end


function solve!(u, t, integs)
    Nu, Np, Nt = size(u)
    dt = t[2] - t[1]

    for i=1:Np
        integ = integs[i].integ
        utmp = integs[i].utmp

        @. utmp = integ.prob.u0
        @. u[:, i, 1] = utmp

        for j=1:Nt-1
            rkstep!(integ, utmp, t[j], dt)
            @. u[:,i,j+1] = utmp
        end
    end
    return nothing
end


Nt = 100
t = range(0.0, 3.0, length=Nt)

Np = 10
a = range(1.0, 3.0, length=Np)

Nu = 2
u0s = [10 * ones(Float64, Nu), 10 * ones(ComplexF64, Nu)]

for u0 in u0s
    uth = zeros(eltype(u0), (Nu, Np, Nt))
    for j=1:Np
    for i=1:Nu
        @. uth[i,j,:] = u0[i] * exp(-a[j] * t)
    end
    end

    probs = Array{Problem}(undef, Np)
    for i=1:Np
        p = (a[i], )
        probs[i] = Problem(func, u0, p)
    end

    u = zeros(eltype(u0), (Nu, Np, Nt))
    for alg in algs
        integs = integrator_ensemble(probs, alg)
        solve!(u, t, integs)

        @test isapprox(u, uth, rtol=1e-5)
        @test allocated(solve!, u, t, integs) == 0
    end
end
