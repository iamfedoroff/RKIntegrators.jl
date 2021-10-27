module RKIntegrators

using Adapt
import CUDA: CuDeviceArray, CuArray, cudaconvert
import LinearAlgebra: dot
import StaticArrays: SVector, SMatrix

export Problem, Integrator, rkstep, rkstep!, integrator_ensemble,
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
struct ExplicitRKIntegrator{U, T, N, L, F, P, K} <: AbstractIntegrator
    func :: F
    u0 :: U
    p :: P
    as :: SMatrix{N, N, T, L}
    bs :: SVector{N, T}
    cs :: SVector{N, T}
    ks :: K
    utmp :: U
end


function Integrator(prob::Problem, alg::ExplicitMethod; kwargs...)
    func, u0, p = prob.func, prob.u0, prob.p

    T = real(eltype(u0))

    as, bs, cs = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)

    ks = zeros(eltype(u0), (size(u0)..., N))
    if typeof(u0) <: CuArray
        ks = CuArray(ks)
    end

    utmp = zero(u0)

    return ExplicitRKIntegrator(func, u0, p, as, bs, cs, ks, utmp)
end


# Out of place -----------------------------------------------------------------
function rkstep(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U, T, N}
    func, p = integ.func, integ.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks

    # i=1:
    ttmp = t + cs[1] * dt
    ks[1] = func(u, p, ttmp)

    @inbounds for i=2:N
        utmp = as[i,1] * ks[1]   # j=1
        for j=2:i-1
            utmp += as[i,j] * ks[j]
        end
        utmp = u + dt * utmp

        ttmp = t + cs[i] * dt
        ks[i] = func(utmp, p, ttmp)
    end

    utmp = u + dt * bs[1] * ks[1]   # i=1
    @inbounds for i=2:N
        utmp += dt * bs[i] * ks[i]
    end
    return utmp
end


# In place ---------------------------------------------------------------------
function rkstep!(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U, T, N}
    func, p = integ.func, integ.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks
    utmp = integ.utmp

    ipre = CartesianIndices(size(utmp))

    # i=1:
    ttmp = t + cs[1] * dt
    @views func(ks[ipre,1], u, p, ttmp)

    @inbounds for i=2:N
        @views @. utmp = as[i,1] * ks[ipre,1]   # j=1
        for j=2:i-1
            @views @. utmp += as[i,j] * ks[ipre,j]
        end
        @. utmp = u + dt * utmp

        ttmp = t + cs[i] * dt
        @views func(ks[ipre,i], utmp, p, ttmp)
    end

    @inbounds for i=1:N
        @views @. u += dt * bs[i] * ks[ipre,i]
    end
    return nothing
end


# In place for CUDA kernels ----------------------------------------------------
# Since the broadcasting does not work for CuDeviceArrays, the loops are written
# explicitly. As a result the step function can be used inside CUDA kernels.
function rkstep!(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U<:CuDeviceArray, T, N}
    func, p = integ.func, integ.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks
    utmp = integ.utmp

    ipre = CartesianIndices(size(utmp))

    # i=1:
    ttmp = t + cs[1] * dt
    @views func(ks[ipre,1], u, p, ttmp)

    @inbounds for i=2:N
        for iu in ipre
            utmp[iu] = as[i,1] * ks[iu,1]   # j=1
            for j=2:i-1
                utmp[iu] += as[i,j] * ks[iu,j]
            end
            utmp[iu] = u[iu] + dt * utmp[iu]
        end

        ttmp = t + cs[i] * dt
        @views func(ks[ipre,i], utmp, p, ttmp)
    end

    @inbounds for i=1:N
        for iu in ipre
            u[iu] += dt * bs[i] * ks[iu,i]
        end
    end
    return nothing
end


# ******************************************************************************
# Embedded integrators
# ******************************************************************************
struct EmbeddedRKIntegrator{U, T, N, L, F, P, K} <: AbstractIntegrator
    func :: F
    u0 :: U
    p :: P
    as :: SMatrix{N, N, T, L}
    bs :: SVector{N, T}
    cs :: SVector{N, T}
    bhats :: SVector{N, T}
    ks :: K
    utmp :: U
    uhat :: U
    atol :: T
    rtol :: T
    edsc :: U
end


function Integrator(
    prob::Problem, alg::EmbeddedMethod; atol=1e-6, rtol=1e-3, kwargs...
)
    func, u0, p = prob.func, prob.u0, prob.p

    T = real(eltype(u0))

    as, bs, cs, bhats = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)
    bhats = SVector{N, T}(bhats)

    ks = zeros(eltype(u0), (size(u0)..., N))
    if typeof(u0) <: CuArray
        ks = CuArray(ks)
    end

    utmp = zero(u0)
    uhat = zero(u0)

    atol = convert(T, atol)
    rtol = convert(T, rtol)
    edsc = zero(u0)

    return EmbeddedRKIntegrator(
        func, u0, p, as, bs, cs, bhats, ks, utmp, uhat, atol, rtol, edsc,
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
    func, p = integ.func, integ.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    atol, rtol = integ.atol, integ.rtol

    err = Inf
    utmp = zero(u)

    while err > 1
        # i=1:
        ttmp = t + cs[1] * dt
        ks[1] = func(u, p, ttmp)

        @inbounds for i=2:N
            utmp = as[i,1] * ks[1]   # j=1
            for j=2:i-1
                utmp += as[i,j] * ks[j]
            end
            utmp = u + dt * utmp

            ttmp = t + cs[i] * dt
            ks[i] = func(utmp, p, ttmp)
        end

        utmp = bs[1] * ks[1]   # i=1
        @inbounds for i=2:N
            utmp += bs[i] * ks[i]
        end
        utmp = u + dt * utmp

        uhat = bhats[1] * ks[1]   # i=1
        @inbounds for i=2:N
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
    func, p = integ.func, integ.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    utmp, uhat = integ.utmp, integ.uhat
    atol, rtol, edsc = integ.atol, integ.rtol, integ.edsc

    ipre = CartesianIndices(size(utmp))

    err = Inf

    while err > 1
        # i=1:
        ttmp = t + cs[1] * dt
        @views func(ks[ipre,1], u, p, ttmp)

        @inbounds for i=2:N
            @views @. utmp = as[i,1] * ks[ipre,1]   # j=1
            for j=2:i-1
                @views @. utmp += as[i,j] * ks[ipre,j]
            end
            @. utmp = u + dt * utmp

            ttmp = t + cs[i] * dt
            @views func(ks[ipre,i], utmp, p, ttmp)
        end

        @views @. utmp = bs[1] * ks[ipre,1]   # i=1
        @inbounds for i=2:N
            @views @. utmp += bs[i] * ks[ipre,i]
        end
        @. utmp = u + dt * utmp

        @views @. uhat = bhats[1] * ks[ipre,1]   # i=1
        @inbounds for i=2:N
            @views @. uhat += bhats[i] * ks[ipre,i]
        end
        @. uhat = u + dt * uhat

        # W.H. Press et al. "Numerical Recipes", 3rd ed. (Cambridge University
        # Press, 2007) p. 913
        #
        # error estimation:
        @. edsc = abs(utmp - uhat) / (atol + max(abs(u), abs(utmp)) * rtol)
        # err = sqrt(sum(abs2, edsc) / length(edsc))
        err = sqrt(real(dot(edsc, edsc)) / length(edsc))
        # use dot product instead of sum to avoid allocations

        # step estimation:
        if err > 1
            rkorder = N - 2   # order of the RK method
            dt = convert(T, 0.9 * dt / err^(1 / rkorder))
        end
    end

    @. u = utmp
    return t + dt
end


# In place for CUDA kernels ----------------------------------------------------
# Since the broadcasting does not work for CuDeviceArrays, the loops are written
# explicitly. As a result the step function can be used inside CUDA kernels.
function substep!(
    integ::EmbeddedRKIntegrator{U, T, N}, u::U, t::T, dt::T,
) where {U<:CuDeviceArray, T, N}
    func, p = integ.func, integ.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    utmp, uhat = integ.utmp, integ.uhat
    atol, rtol, edsc = integ.atol, integ.rtol, integ.edsc

    ipre = CartesianIndices(size(utmp))

    err = Inf

    while err > 1
        # i=1:
        ttmp = t + cs[1] * dt
        @views func(ks[ipre,1], u, p, ttmp)

        @inbounds for i=2:N
            for iu in ipre
                utmp[iu] = as[i,1] * ks[iu,1]   # j=1
                for j=2:i-1
                    utmp[iu] += as[i,j] * ks[iu,j]
                end
                utmp[iu] = u[iu] + dt * utmp[iu]
            end

            ttmp = t + cs[i] * dt
            @views func(ks[ipre,i], utmp, p, ttmp)
        end

        @inbounds for iu in ipre
            utmp[iu] = bs[1] * ks[iu,1]   # i=1
            for i=2:N
                utmp[iu] += bs[i] * ks[iu,i]
            end
            utmp[iu] = u[iu] + dt * utmp[iu]
        end

        @inbounds for iu in ipre
            uhat[iu] = bhats[1] * ks[iu,1]   # i=1
            for i=1:N
                uhat[iu] += bhats[i] * ks[iu,i]
            end
            uhat[iu] = u[iu] + dt * uhat[iu]
        end

        # W.H. Press et al. "Numerical Recipes", 3rd ed. (Cambridge University
        # Press, 2007) p. 913
        #
        # error estimation:
        @inbounds for iu in ipre
            edsc[iu] = abs(utmp[iu] - uhat[iu]) /
                       (atol + max(abs(u[iu]), abs(utmp[iu])) * rtol)
            # @. edsc = abs(utmp - uhat) / (atol + max(abs(u), abs(utmp)) * rtol)
        end

        err = zero(T)
        @inbounds for iu in ipre
            err += real(edsc[iu]^2)   # err = sqrt(sum(abs2, edsc) / length(edsc))
        end
        err = sqrt(err / length(edsc))

        # step estimation:
        if err > 1
            rkorder = N - 2   # order of the RK method
            dt = convert(T, 0.9 * dt / err^(1 / rkorder))
        end
    end

    @inbounds for iu in ipre
        u[iu] = utmp[iu]   # @. u = utmp
    end

    return t + dt
end


# ******************************************************************************
# Ensemble integrators
# ******************************************************************************
struct EnsembleIntegrator{I, U} <: AbstractIntegrator
    integ :: I
    utmp :: U
end


function integrator_ensemble(probs::Vector{Problem}, alg)
    Np = length(probs)
    integs = Array{EnsembleIntegrator}(undef, Np)
    for i=1:Np
        prob = probs[i]
        integ = Integrator(prob, alg)
        utmp = zero(prob.u0)
        integs[i] = EnsembleIntegrator(integ, utmp)
    end
    return integs
end


# ******************************************************************************
# CUDA Adaptors
# ******************************************************************************
function Adapt.adapt_structure(to, integ::ExplicitRKIntegrator)
    ks = adapt(CuArray, integ.ks)
    return ExplicitRKIntegrator(
        adapt(to, integ.func),
        adapt(to, integ.u0),
        adapt(to, integ.p),
        adapt(to, integ.as),
        adapt(to, integ.bs),
        adapt(to, integ.cs),
        adapt(to, ks),
        adapt(to, integ.utmp),
    )
end


function Adapt.adapt_structure(to, integ::EmbeddedRKIntegrator)
    ks = adapt(CuArray, integ.ks)
    return EmbeddedRKIntegrator(
        adapt(to, integ.func),
        adapt(to, integ.u0),
        adapt(to, integ.p),
        adapt(to, integ.as),
        adapt(to, integ.bs),
        adapt(to, integ.cs),
        adapt(to, integ.bhats),
        adapt(to, ks),
        adapt(to, integ.utmp),
        adapt(to, integ.uhat),
        adapt(to, integ.atol),
        adapt(to, integ.rtol),
        adapt(to, integ.edsc),
    )
end


function Adapt.adapt_structure(to, einteg::EnsembleIntegrator)
    return EnsembleIntegrator(
        adapt(to, einteg.integ),
        adapt(to, einteg.utmp),
    )
end


end
