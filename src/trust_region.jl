"""Functions & structs for trust-region handling"""

"""Trust-region struct"""
mutable struct TrustRegions
    Δ::Matrix{Float64}
    Δ_type::Symbol
end


function Base.show(io::IO, tr::TrustRegions)
    println(io, "Trust-region struct")
    @printf("   Trust-region type   : %s\n", tr.Δ_type)
    @printf("   Trust-region size Δ : %1.2e\n", tr.Δ[1,1])
end


"""Initialize trust-region struct from a single scalar"""
function TrustRegions(nx::Int, N::Int, Δ0::Float64)
    return TrustRegions(Δ0 * ones(nx, N), :Uniform)
end


"""Initialize trust-region struct from a vector of length `nx`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Vector{Float64})
    @assert length(Δ0s) == nx "Length of Δ0s must match number of states"
    return TrustRegions(repeat(Δ0s, 1, N), :NodeDependent)
end


"""Initialize trust-region struct from a matrix of size `(nx, N)`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Matrix{Float64}, x_dependent::Bool = false)
    @assert size(Δ0s) == (nx, N) "Size of Δ0s must match number of states and number of nodes"
    if x_dependent
        Δ_type = :StateDependent
    else
        Δ_type = :Independent
    end
    return TrustRegions(Δ0s, Δ_type)
end



"""
    update_trust_region!(algo::SCPAlgorithm, action::Symbol)

Apply a trust-region update `action ∈ (:shrink, :expand, :keep)` to `algo.tr` (and to
`algo.tr_u` when the control trust-region is used), clamped to `algo.Δ_bounds`.
Returns `true` if the trust-region size actually changed.
"""
function update_trust_region!(algo::SCPAlgorithm, action::Symbol)
    @assert action in (:shrink, :expand, :keep) "Invalid trust-region action: $action"
    action == :keep && return false

    function _update(Δ)
        if action == :shrink
            return max.(Δ / algo.alphas[1], algo.Δ_bounds[1])
        else
            return min.(Δ * algo.alphas[2], algo.Δ_bounds[2])
        end
    end

    Δ_new = _update(algo.tr.Δ)
    changed = Δ_new != algo.tr.Δ
    algo.tr.Δ = Δ_new

    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        Δu_new = _update(algo.tr_u.Δ)
        changed = changed || (Δu_new != algo.tr_u.Δ)
        algo.tr_u.Δ = Δu_new
    end
    return changed
end


"""
    update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)

Update trust-region size Δ from the ratio `rho_i`: shrink if `rho_i < rhos[2]`, expand if
`rho_i >= rhos[3]`, keep otherwise.
Returns `true` if the trust-region size actually changed.
"""
function update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)
    return update_trust_region!(algo, trust_region_action(algo, rho_i))
end


"""Trust-region action (`:shrink`, `:expand` or `:keep`) for a ratio `rho_i`"""
function trust_region_action(algo::SCPAlgorithm, rho_i::Real)
    if rho_i < algo.rhos[2]
        return :shrink
    elseif rho_i >= algo.rhos[3]
        return :expand
    else
        return :keep
    end
end


"""Whether every trust-region radius (state, and control if used) is at its lower bound `Δ_bounds[1]`"""
function trust_region_at_floor(algo::SCPAlgorithm)
    at_floor = all(algo.tr.Δ .<= algo.Δ_bounds[1])
    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        at_floor = at_floor && all(algo.tr_u.Δ .<= algo.Δ_bounds[1])
    end
    return at_floor
end


"""
    clamp_trust_region!(algo::SCPAlgorithm)

Clamp the trust-region radii into `algo.Δ_bounds` (e.g. an initial `Δ0` below `Δ_min`, where the
subproblem solution is below the solver accuracy and the trust-region tests are meaningless).
"""
function clamp_trust_region!(algo::SCPAlgorithm)
    algo.tr.Δ = clamp.(algo.tr.Δ, algo.Δ_bounds[1], algo.Δ_bounds[2])
    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        algo.tr_u.Δ = clamp.(algo.tr_u.Δ, algo.Δ_bounds[1], algo.Δ_bounds[2])
    end
    return
end


"""
    ensure_trust_region_control!(algo::SCPAlgorithm, prob::OptimalControlProblem)

If the control trust-region is requested (`use_trustregion_control = true`) but was not built
at construction (because `nu` was not passed), build it now from `prob.nu` and `algo.Δ0_u`.
"""
function ensure_trust_region_control!(algo::SCPAlgorithm, prob::OptimalControlProblem)
    if hasproperty(algo, :use_trustregion_control) && algo.use_trustregion_control && isnothing(algo.tr_u)
        algo.tr_u = TrustRegions(prob.nu, prob.N, algo.Δ0_u)
    end
    return
end


"""
    trust_region_active(algo, x, u, x_ref, u_ref; rtol = 1e-3, atol = 1e-8)

Whether the convex subproblem solution `(x,u)` lies on the boundary of the trust-region
around `(x_ref,u_ref)`, i.e. some component satisfies `|x - x_ref| ≥ max((1 - rtol) Δ - atol, Δ/2)`
(`atol` accounts for the absolute accuracy of interior-point solutions at small Δ).
When the trust-region is inactive, the subproblem solution is the minimizer of the convex
model over the convex feasible set, so small model/merit changes indicate stationarity;
when it is active, they may only reflect a small trust-region.
"""
function trust_region_active(algo::SCPAlgorithm, x, u, x_ref, u_ref; rtol::Real = 1e-3, atol::Real = 1e-8)
    _threshold(Δ) = max.((1 - rtol) .* Δ .- atol, Δ ./ 2)
    nk = size(x, 2)
    if any(abs.(x - x_ref) .>= _threshold(algo.tr.Δ[:, 1:nk]))
        return true
    end
    if hasproperty(algo, :use_trustregion_control) && algo.use_trustregion_control && !isnothing(algo.tr_u)
        nku = size(u, 2)
        if any(abs.(u - u_ref) .>= _threshold(algo.tr_u.Δ[:, 1:nku]))
            return true
        end
    end
    return false
end


"""
    reference_convex_feasible(prob, x_ref, u_ref; atol = 1e-6)

Check whether `(x_ref, u_ref)` satisfies the convex constraints of `prob.model` that involve
only `x` and `u` (constraints that involve other variables are skipped).
Must be called before the linearized/trust-region constraints are added to the model.
Returns `true` if the check cannot be carried out.
"""
function reference_convex_feasible(prob::OptimalControlProblem, x_ref, u_ref; atol::Real = 1e-6)
    x = prob.model[:x]
    u = prob.model[:u]
    if size(x) != size(x_ref) || size(u) != size(u_ref)
        return true
    end
    point = Dict{VariableRef,Float64}()
    for I in CartesianIndices(x)
        point[x[I]] = x_ref[I]
    end
    for I in CartesianIndices(u)
        point[u[I]] = u_ref[I]
    end
    try
        report = primal_feasibility_report(prob.model, point; atol = atol, skip_missing = true)
        return isempty(report)
    catch err
        @debug "Could not check convex feasibility of the reference: $err"
        return true
    end
end


"""
    classify_step(ΔJ, ΔL, J_ref, ref_feasible, ref_is_initial_guess, tr_active;
                  tol_ΔL = 1e-8, tol_abs = Inf)

Classify a trust-region step from the actual (`ΔJ`) and predicted (`ΔL`) merit reductions.
Returns `(rho, kind)`, where `rho = ΔJ/ΔL` (or `NaN` if `ΔL == 0`) and `kind` is one of:

- `:restoration`: the reference does not satisfy the convex constraints (`ref_feasible = false`,
  or `ΔL < -ε` while the reference is still the initial guess, which indicates it: for a
  convex-feasible `z̄` and a consistent model, `L(z̄) = J(z̄)` and `ΔL ≥ 0`). The ratio is
  meaningless; the step is accepted without a ratio test.
- `:ratio`: `ΔL > ε`; the regular ratio test applies to `rho`.
- `:stationary`: `ΔL ≤ ε` and the trust-region is inactive at the candidate (null step). The
  reference is feasible for its own subproblem with `L(z̄) = J(z̄)`, so the subproblem cannot
  improve on it by more than `ε` (a negative `ΔL` only means the solver returned a point worse
  than `z̄`, i.e. `z̄` is optimal to within the solver accuracy); since the trust-region is
  inactive, `z̄` is a stationary point of the penalized problem.
- `:unresolved`: `|ΔL| ≤ ε` while the step is bounded by the trust-region: the predicted decrease
  is below resolution because the trust-region is too small to tell. The reference is kept and
  the trust-region is expanded (shrinking would only make `ΔL` smaller).
- `:reject`: `ΔL < -ε` while the step is bounded by the trust-region (inconsistent model or
  solver inaccuracy); the trust-region is shrunk.

The threshold is `ε = min(tol_ΔL * max(1, |J_ref|), tol_abs)`: relative to the merit to
account for solver accuracy, but never above `tol_abs` (the optimality tolerance), so that a
constant offset in the objective cannot hide a decrease the stopping test would care about.
"""
function classify_step(
    ΔJ::Real, ΔL::Real, J_ref::Real,
    ref_feasible::Bool, ref_is_initial_guess::Bool, tr_active::Bool;
    tol_ΔL::Real = 1e-8,
    tol_abs::Real = Inf,
)
    ε = min(tol_ΔL * max(1.0, abs(J_ref)), tol_abs)
    rho = ΔL == 0 ? NaN : ΔJ / ΔL
    if !ref_feasible || (ref_is_initial_guess && ΔL < -ε)
        return rho, :restoration
    elseif ΔL > ε
        return rho, :ratio
    elseif !tr_active
        return rho, :stationary
    elseif ΔL >= -ε
        return rho, :unresolved
    else
        return rho, :reject
    end
end


"""Whether every trust-region radius (state, and control if used) is at its upper bound `Δ_bounds[2]`"""
function trust_region_at_ceiling(algo::SCPAlgorithm)
    at_ceiling = all(algo.tr.Δ .>= algo.Δ_bounds[2])
    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        at_ceiling = at_ceiling && all(algo.tr_u.Δ .>= algo.Δ_bounds[2])
    end
    return at_ceiling
end


"""
Trust-region action for a classified step: ratio-based for `:ratio`, expand for `:unresolved`,
shrink for `:reject`, keep otherwise
"""
function trust_region_action(algo::SCPAlgorithm, step_kind::Symbol, rho_i::Real)
    if step_kind == :ratio
        return trust_region_action(algo, rho_i)
    elseif step_kind == :unresolved
        return :expand
    elseif step_kind == :reject
        return :shrink
    else
        return :keep
    end
end


"""
    set_trust_region_constraints!(
        algo::SCPAlgorithm,
        prob::OptimalControlProblem,
        x_ref::Union{Matrix,Adjoint},
        u_ref::Union{Matrix,Adjoint},
    )
    
Set trust-region constraints to JuMP model in `prob`
"""
function set_trust_region_constraints!(
    algo::SCPAlgorithm,
    prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint},
)
    ensure_trust_region_control!(algo, prob)

    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:size(x_ref,2)],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.tr.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:size(x_ref,2)],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.tr.Δ[:,k])

    if algo.use_trustregion_control
        @constraint(prob.model, constraint_trust_region_u_lb[k in 1:size(u_ref,2)],
            -(prob.model[:u][:,k] - u_ref[:,k]) <= algo.tr_u.Δ[:,k])
        @constraint(prob.model, constraint_trust_region_u_ub[k in 1:size(u_ref,2)],
            prob.model[:u][:,k] - u_ref[:,k]  <= algo.tr_u.Δ[:,k])
    end
    return
end
