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
    prob :: Problem{F, U, P}
    as :: SMatrix{N, N, T, L}
    bs :: SVector{N, T}
    cs :: SVector{N, T}
    ks :: K
    utmp :: U
end


function Integrator(prob::Problem, alg::ExplicitMethod; kwargs...)
    T = real(eltype(prob.u0))

    as, bs, cs = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)

    ks = [zero(prob.u0) for i in 1:N]

    utmp = zero(prob.u0)

    return ExplicitRKIntegrator(prob, as, bs, cs, ks, utmp)
end


# Out of place -----------------------------------------------------------------
function rkstep(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks

    @inbounds for i=1:N
        utmp = zero(u)
        for j=1:i-1
            utmp += as[i,j] * ks[j]
        end
        utmp = u + dt * utmp
        ttmp = t + cs[i] * dt
        ks[i] = func(utmp, p, ttmp)
    end

    utmp = u
    @inbounds for i=1:N
        utmp += dt * bs[i] * ks[i]
    end
    return utmp
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


# In place for CUDA kernels ----------------------------------------------------
# Since the broadcasting does not work for CuDeviceArrays, the loops are written
# explicitly. As a result the step function can be used inside CUDA kernels.
function rkstep!(
    integ::ExplicitRKIntegrator{U, T, N}, u::U, t::T, dt::T
) where {U<:CuDeviceArray, T, N}
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, ks = integ.as, integ.bs, integ.cs, integ.ks
    utmp = integ.utmp

    Nu = length(u)

    @inbounds for i=1:N
        for iu=1:Nu
            utmp[iu] = 0   # @. utmp = 0
        end

        for j=1:i-1
            for iu=1:Nu
                utmp[iu] += as[i,j] * ks[j][iu]   # @. utmp += as[i,j] * ks[j]
            end
        end

        for iu=1:Nu
            utmp[iu] = u[iu] + dt * utmp[iu]   # @. utmp = u + dt * utmp
        end

        ttmp = t + cs[i] * dt
        func(ks[i], utmp, p, ttmp)
    end

    @inbounds for i=1:N
        for iu=1:Nu
            u[iu] += dt * bs[i] * ks[i][iu]   # @. u += dt * bs[i] * ks[i]
        end
    end
    return nothing
end


# ******************************************************************************
# Embedded integrators
# ******************************************************************************
struct EmbeddedRKIntegrator{U, T, N, L, F, P, K} <: AbstractIntegrator
    prob :: Problem{F, U, P}
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
    T = real(eltype(prob.u0))

    as, bs, cs, bhats = tableau(alg)
    N = length(cs)
    as = SMatrix{N, N, T}(as)
    bs = SVector{N, T}(bs)
    cs = SVector{N, T}(cs)
    bhats = SVector{N, T}(bhats)

    ks = [zero(prob.u0) for i in 1:N]

    utmp = zero(prob.u0)
    uhat = zero(prob.u0)

    atol = convert(T, atol)
    rtol = convert(T, rtol)
    edsc = zero(prob.u0)

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
    func, p = integ.prob.func, integ.prob.p
    as, bs, cs, bhats, ks = integ.as, integ.bs, integ.cs, integ.bhats, integ.ks
    utmp, uhat = integ.utmp, integ.uhat
    atol, rtol, edsc = integ.atol, integ.rtol, integ.edsc

    Nu = length(u)

    err = Inf

    while err > 1
        @inbounds for i=1:N
            for iu=1:Nu
                utmp[iu] = 0   # @. utmp = 0
            end

            for j=1:i-1
                for iu=1:Nu
                    utmp[iu] += as[i,j] * ks[j][iu]   # @. utmp += as[i,j] * ks[j]
                end
            end

            for iu=1:Nu
                utmp[iu] = u[iu] + dt * utmp[iu]   # @. utmp = u + dt * utmp
            end

            ttmp = t + cs[i] * dt
            func(ks[i], utmp, p, ttmp)
        end


        @inbounds for iu=1:Nu
            utmp[iu] = 0   # @. utmp = 0
        end
        @inbounds for i=1:N
            for iu=1:Nu
                utmp[iu] += bs[i] * ks[i][iu]   # @. utmp += bs[i] * ks[i]
            end
        end
        @inbounds for iu=1:Nu
            utmp[iu] = u[iu] + dt * utmp[iu]   # @. utmp = u + dt * utmp
        end


        @inbounds for iu=1:Nu
            uhat[iu] = 0   # @. uhat = 0
        end
        @inbounds for i=1:N
            for iu=1:Nu
                uhat[iu] += bhats[i] * ks[i][iu]   # @. uhat += bhats[i] * ks[i]
            end
        end
        @inbounds for iu=1:Nu
            uhat[iu] = u[iu] + dt * uhat[iu]   # @. uhat = u + dt * uhat
        end

        # W.H. Press et al. "Numerical Recipes", 3rd ed. (Cambridge University
        # Press, 2007) p. 913
        #
        # error estimation:
        @inbounds for iu=1:Nu
            edsc[iu] = abs(utmp[iu] - uhat[iu]) /
                       (atol + max(abs(u[iu]), abs(utmp[iu])) * rtol)
            # @. edsc = abs(utmp - uhat) / (atol + max(abs(u), abs(utmp)) * rtol)
        end

        err = zero(T)
        @inbounds for iu=1:Nu
            err += real(edsc[iu]^2)   # err = sqrt(sum(abs2, edsc) / length(edsc))
        end
        err = sqrt(err) / Nu

        # step estimation:
        if err > 1
            rkorder = N - 2   # order of the RK method
            dt = convert(T, 0.9 * dt / err^(1 / rkorder))
        end
    end

    @inbounds for iu=1:Nu
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
function Adapt.adapt_structure(to, prob::Problem)
    return Problem(
        adapt(to, prob.func),
        adapt(to, prob.u0),
        adapt(to, prob.p),
    )
end


function Adapt.adapt_structure(to, integ::ExplicitRKIntegrator)
    if eltype(integ.ks) <: AbstractArray
        ks = SVector{length(integ.ks)}(cudaconvert.(integ.ks))
    else
        ks = adapt(CuArray, integ.ks)
    end
    return ExplicitRKIntegrator(
        adapt(to, integ.prob),
        adapt(to, integ.as),
        adapt(to, integ.bs),
        adapt(to, integ.cs),
        adapt(to, ks),
        adapt(to, integ.utmp),
    )
end


function Adapt.adapt_structure(to, integ::EmbeddedRKIntegrator)
    if eltype(integ.ks) <: AbstractArray
        ks = SVector{length(integ.ks)}(cudaconvert.(integ.ks))
    else
        ks = adapt(CuArray, integ.ks)
    end
    return EmbeddedRKIntegrator(
        adapt(to, integ.prob),
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
