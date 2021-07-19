module RKIntegrators

import StaticArrays: SVector, SMatrix

export Problem, Integrator, rkstep, rkstep!,
       RK3, SSPRK3, RK4, Tsit5,   # explicit methods
       RK45, DP5, ATsit5   # embedded methods


include("methods.jl")


struct Problem{F, U, P}
    func :: F
    u0 :: U
    p :: P
end


abstract type AbstractIntegrator end


# ******************************************************************************
# Explicit integrators
# ******************************************************************************
struct ExplicitRKIntegrator{U, T, N, L, F, P} <: AbstractIntegrator
    prob :: Problem{F, U, P}
    as :: SMatrix{N, N, T, L}
    bs :: SVector{N, T}
    cs :: SVector{N, T}
    ks :: Union{Vector{U}, SVector{N, U}}
    utmp :: U
end


function Integrator(prob::Problem, alg::ExplicitMethod; kwargs...)
    u0 = prob.u0
    U = typeof(u0)
    T = real(eltype(u0))

    as, bs, cs = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)

    if U <: AbstractArray
        ks = SVector{N, U}([zero(u0) for i in 1:N])
    else
        ks = [zero(u0) for i in 1:N]
    end

    utmp = zero(u0)

    return ExplicitRKIntegrator(prob, as, bs, cs, ks, utmp)
end


# Out of place -----------------------------------------------------------------
function rkstep(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks

    @inbounds for i=1:N
        utmp = zero(T)
        for j=1:i-1
            utmp += as[i,j] * ks[j]
        end
        utmp = u + dt * utmp
        ttmp = t + cs[i] * dt
        ks[i] = func(utmp, p, ttmp)
    end

    @inbounds for i=1:N
        u += dt * bs[i] * ks[i]
    end
    return u
end


# In place ---------------------------------------------------------------------
function rkstep!(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks
    utmp = integ.utmp

    @inbounds for i=1:N
        @. utmp = 0
        for j=1:i-1
            @. utmp += as[i,j] * ks[j]
        end
        @. utmp = u + dt * utmp
        ttmp = t + cs[i] * dt
        func(ks[i], utmp, p, ttmp)
    end

    @inbounds for i=1:N
        @. u += dt * bs[i] * ks[i]
    end
    return nothing
end


# ******************************************************************************
# Embedded integrators
# ******************************************************************************
struct EmbeddedRKIntegrator{U, T, N, L, F, P} <: AbstractIntegrator
    prob :: Problem{F, U, P}
    as :: SMatrix{N, N, T, L}
    bs :: SVector{N, T}
    cs :: SVector{N, T}
    bhats :: SVector{N, T}
    ks :: Union{Vector{U}, SVector{N, U}}
    utmp :: U
    uhat :: U
    atol :: T
    rtol :: T
    edsc :: U
end


function Integrator(
    prob::Problem, alg::EmbeddedMethod; atol=1e-6, rtol=1e-3, kwargs...
)
    u0 = prob.u0
    U = typeof(u0)
    T = real(eltype(u0))

    as, bs, cs, bhats = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)
    bhats = SVector{N, T}(bhats)

    if U <: AbstractArray
        ks = SVector{N, U}([zero(u0) for i in 1:N])
    else
        ks = [zero(u0) for i in 1:N]
    end

    utmp = zero(u0)
    uhat = zero(u0)

    atol = convert(T, atol)
    rtol = convert(T, rtol)
    edsc = zero(u0)

    return EmbeddedRKIntegrator(
        prob, as, bs, cs, bhats, ks, utmp, uhat, atol, rtol, edsc,
    )
end


# Out of place -----------------------------------------------------------------
function rkstep(
    integ::EmbeddedRKIntegrator{U, T, N}, u::U, t::T, dt::T,
) where {U, T, N}
    tend = t + dt

    usub, tsub = substep(integ, u, t, dt)

    while tsub < tend
        dtsub = tend - tsub
        usub, tsub = substep(integ, usub, tsub, dtsub)
    end

    return usub
end


function substep(
    integ::EmbeddedRKIntegrator{U, T, N}, u::U, t::T, dt::T,
) where {U, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    atol, rtol = integ.atol, integ.rtol

    err = Inf
    utmp = zero(u)

    while err > 1
        @inbounds for i=1:N
            utmp = zero(T)
            for j=1:i-1
                utmp += as[i,j] * ks[j]
            end
            utmp = u + dt * utmp
            ttmp = t + cs[i] * dt
            ks[i] = func(utmp, p, ttmp)
        end

        utmp = zero(T)
        @inbounds for i=1:N
            utmp += bs[i] * ks[i]
        end
        utmp = u + dt * utmp

        uhat = zero(T)
        @inbounds for i=1:N
            uhat += bhats[i] * ks[i]
        end
        uhat = u + dt * uhat

        # W.H. Press et al. "Numerical Recipes", 3rd ed. (Cambridge University
        # Press, 2007) p. 913
        #
        # error estimation:
        D = abs(utmp - uhat)
        scale = atol + max(abs(u), abs(utmp)) * rtol
        err = (D / scale)^2

        # step estimation:
        if err > 1
            rkorder = N - 2   # order of the RK method
            dt = convert(T, 0.9 * dt / err^(1 / rkorder))
        end
    end

    return utmp, t + dt
end


# In place ---------------------------------------------------------------------
function rkstep!(
    integ::EmbeddedRKIntegrator{U, T, N}, u::U, t::T, dt::T,
) where {U, T, N}
    tend = t + dt

    tsub = substep!(integ, u, t, dt)

    while tsub < tend
        dtsub = tend - tsub
        tsub = substep!(integ, u, tsub, dtsub)
    end

    return nothing
end


function substep!(
    integ::EmbeddedRKIntegrator{U, T, N}, u::U, t::T, dt::T,
) where {U, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    utmp, uhat = integ.utmp, integ.uhat
    atol, rtol, edsc = integ.atol, integ.rtol, integ.edsc

    err = Inf

    while err > 1
        @inbounds for i=1:N
            @. utmp = 0
            for j=1:i-1
                @. utmp += as[i,j] * ks[j]
            end
            @. utmp = u + dt * utmp
            ttmp = t + cs[i] * dt
            func(ks[i], utmp, p, ttmp)
        end

        @. utmp = 0
        @inbounds for i=1:N
            @. utmp += bs[i] * ks[i]
        end
        @. utmp = u + dt * utmp

        @. uhat = 0
        @inbounds for i=1:N
            @. uhat += bhats[i] * ks[i]
        end
        @. uhat = u + dt * uhat

        # W.H. Press et al. "Numerical Recipes", 3rd ed. (Cambridge University
        # Press, 2007) p. 913
        #
        # error estimation:
        @. edsc = (abs(utmp - uhat) / (atol + max(abs(u), abs(utmp)) * rtol))^2
        err = sqrt(sum(abs, edsc) / length(edsc))

        # step estimation:
        if err > 1
            rkorder = N - 2   # order of the RK method
            dt = convert(T, 0.9 * dt / err^(1 / rkorder))
        end
    end

    @. u = utmp
    return t + dt
end


end
