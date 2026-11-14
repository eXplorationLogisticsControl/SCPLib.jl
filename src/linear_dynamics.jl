"""Linear dynamics: freeze the dynamics linearization so that `solve!` does no ODE propagation"""


"""
    linear_dynamics_defects(Φ_A, Φ_B, Φ_c, x, u)

Defects `x[:,k+1] - (Φ_A[:,:,k] x[:,k] + Φ_B[:,:,k] u[:,k] + Φ_c[:,k])`, `k = 1:N-1`, of the
discrete-time affine dynamics `(Φ_A, Φ_B, Φ_c)`.
"""
function linear_dynamics_defects(Φ_A::AbstractArray{<:Real,3}, Φ_B::AbstractArray{<:Real,3}, Φ_c::AbstractMatrix, x, u)
    nx, _, Nseg = size(Φ_A)
    g = zeros(promote_type(eltype(x), eltype(u), Float64), nx, Nseg)
    for k in 1:Nseg
        @views g[:, k] .= x[:, k+1] .- Φ_A[:, :, k] * x[:, k] .- Φ_B[:, :, k] * u[:, k] .- Φ_c[:, k]
    end
    return g
end


"""Frozen affine dynamics `x[:,k+1] = Φ_A[:,:,k] x[:,k] + Φ_B[:,:,k] u[:,k] + Φ_c[:,k]`"""
struct FrozenLinearDynamics
    Φ_A::Array{Float64,3}
    Φ_B::Array{Float64,3}
    Φ_c::Matrix{Float64}
end


"""`prob.set_dynamics_cache!` hook installed by `set_linear_dynamics!`: loads the frozen matrices, returns the defects"""
struct LinearDynamicsCacheHook <: Function
    dyn::FrozenLinearDynamics
end

function (hook::LinearDynamicsCacheHook)(prob, x, u)
    prob.lincache.Φ_A .= hook.dyn.Φ_A
    prob.lincache.Φ_B .= hook.dyn.Φ_B
    prob.lincache.Φ_c .= hook.dyn.Φ_c
    return linear_dynamics_defects(hook.dyn.Φ_A, hook.dyn.Φ_B, hook.dyn.Φ_c, x, u)
end


"""`prob.fun_get_trajectory` hook installed by `set_linear_dynamics!`: returns `(nothing, defects)`"""
struct LinearDynamicsTrajectoryHook <: Function
    dyn::FrozenLinearDynamics
end

function (hook::LinearDynamicsTrajectoryHook)(prob, x, u)
    return nothing, linear_dynamics_defects(hook.dyn.Φ_A, hook.dyn.Φ_B, hook.dyn.Φ_c, x, u)
end


"""
    set_linear_dynamics!(prob; Φ_A = nothing, Φ_B = nothing, Φ_c = nothing, x_ref = nothing, u_ref = nothing)

Declare that the dynamics of `prob` are linear (affine), so that their multiple-shooting
linearization `x[:,k+1] = Φ_A[:,:,k] x[:,k] + Φ_B[:,:,k] u[:,k] + Φ_c[:,k]` is exact and does
not depend on the reference. The matrices are computed (or taken) once; afterwards `solve!`
does not propagate any ODE: `prob.set_dynamics_cache!` only copies them into `prob.lincache`,
and `prob.fun_get_trajectory` evaluates the dynamics defects algebraically. Non-convex path and
boundary constraints are given as usual through `g_noncvx`/`h_noncvx`.

Two ways to provide the matrices:

- Pass none of `Φ_A`, `Φ_B`, `Φ_c`: the dynamics `prob.eom!` (continuous-time, e.g.
  `ẋ = A(t)x + B(t)u + c(t)`, with `u` held constant on each segment) are discretized once with
  the default ODE propagation, about `(x_ref, u_ref)` (zeros by default; the result does not depend
  on the reference when the dynamics are linear). `u_bias` is included in `Φ_c`.
  For an `ImpulsiveProblem`, `dfdu` may depend on `t` only (not on `x` or `u`).
- Pass all three: discrete-time matrices of size `(nx,nx,N-1)`, `(nx,nu,N-1)`, `(nx,N-1)`.
  They are used as given (`u_bias` is not applied: include `Φ_B[:,:,k] * u_bias[:,k]` in `Φ_c`).
  `prob.eom!` is then never called by `solve!`.

Supported for `ContinuousProblem` and `ImpulsiveProblem` with `shooting_method = :multiple`.
The dynamics are not checked for linearity: with nonlinear `eom!`, the frozen matrices are the
linearization about `(x_ref, u_ref)` only, and the solution does not satisfy `eom!`.

Overwrites `prob.set_dynamics_cache!` and `prob.fun_get_trajectory` (with a warning if they were
user-defined). A user `set_linearized_constraints!` override must call `prob.set_dynamics_cache!`
(or use `prob.lincache`) to see the matrices.

`get_trajectory(prob, x, u)` still propagates `prob.eom!`; use
`SCPLib.get_dynamics_defects(prob, x, u)` to evaluate the defects of the frozen dynamics.

Returns `prob`.
"""
function set_linear_dynamics!(
    prob::Union{ContinuousProblem,ImpulsiveProblem};
    Φ_A::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
    Φ_B::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
    Φ_c::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    x_ref::Union{Nothing,AbstractMatrix} = nothing,
    u_ref::Union{Nothing,AbstractMatrix} = nothing,
)
    @assert prob.shooting_method == :multiple "set_linear_dynamics! requires shooting_method = :multiple"
    @assert prob.lincache isa MultipleShootingCache "set_linear_dynamics! requires a MultipleShootingCache"
    nx, nu, N = prob.nx, prob.nu, prob.N

    given = (!isnothing(Φ_A), !isnothing(Φ_B), !isnothing(Φ_c))
    @assert all(given) || !any(given) "Pass all of Φ_A, Φ_B, Φ_c, or none of them"
    if all(given)
        @assert size(Φ_A) == (nx, nx, N-1) "Φ_A must be of size (nx, nx, N-1) = $((nx, nx, N-1))"
        @assert size(Φ_B) == (nx, nu, N-1) "Φ_B must be of size (nx, nu, N-1) = $((nx, nu, N-1))"
        @assert size(Φ_c) == (nx, N-1)     "Φ_c must be of size (nx, N-1) = $((nx, N-1))"
        if !isnothing(x_ref) || !isnothing(u_ref)
            @warn "x_ref and u_ref are ignored when Φ_A, Φ_B, Φ_c are given"
        end
        if any(!iszero, prob.u_bias)
            @warn "prob.u_bias is not applied to user-supplied Φ_A, Φ_B, Φ_c: include Φ_B[:,:,k] * u_bias[:,k] in Φ_c"
        end
    else
        @assert isnothing(x_ref) || size(x_ref) == (nx, N) "x_ref must be of size (nx, N) = $((nx, N))"
        @assert isnothing(u_ref) || size(u_ref) == (nu, prob.Nu) "u_ref must be of size (nu, Nu) = $((nu, prob.Nu))"
    end
    user_hooks = (prob.set_dynamics_cache!, prob.fun_get_trajectory)
    if any(h -> !isnothing(h) && !(h isa Union{LinearDynamicsCacheHook,LinearDynamicsTrajectoryHook}), user_hooks)
        @warn "set_linear_dynamics! overwrites the user-defined set_dynamics_cache! / fun_get_trajectory"
    end

    if all(given)
        # `Array{Float64}(⋅)` copies, so later changes to the caller's arrays have no effect
        dyn = FrozenLinearDynamics(Array{Float64,3}(Φ_A), Array{Float64,3}(Φ_B), Matrix{Float64}(Φ_c))
    else
        # discretize once with the default (ODE) implementation, which dispatches on the problem type
        x_disc = isnothing(x_ref) ? zeros(nx, N) : Matrix{Float64}(x_ref)
        u_disc = isnothing(u_ref) ? zeros(nu, prob.Nu) : Matrix{Float64}(u_ref)
        set_dynamics_cache!(prob, x_disc, u_disc)
        dyn = FrozenLinearDynamics(copy(prob.lincache.Φ_A), copy(prob.lincache.Φ_B), copy(prob.lincache.Φ_c))
    end

    prob.set_dynamics_cache! = LinearDynamicsCacheHook(dyn)
    prob.fun_get_trajectory = LinearDynamicsTrajectoryHook(dyn)
    prob.set_dynamics_cache!(prob, zeros(nx, N), zeros(nu, prob.Nu))     # load the matrices into prob.lincache
    return prob
end
