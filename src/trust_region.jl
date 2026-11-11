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
Scale trust-region radii. `NaN` entries are copied through and are not used as radii.
Mixed problems mark the unused continuous column `N` this way.
"""
function scale_trust_radii(Δ::AbstractMatrix, factor::Real, bound::Real, shrink::Bool)
    out = similar(Δ)
    for i in eachindex(Δ)
        v = Δ[i]
        if isnan(v)
            out[i] = v
        elseif shrink
            out[i] = max(v / factor, bound)
        else
            out[i] = min(v * factor, bound)
        end
    end
    return out
end


"""Whether every non-`NaN` radius is at or below `Δ_min` (`NaN` radii are unused slots)"""
radii_at_lower_bound(Δ::AbstractMatrix, Δ_min::Real) = all(v -> isnan(v) || v <= Δ_min, Δ)


"""Whether every non-`NaN` radius is at or above `Δ_max` (`NaN` radii are unused slots)"""
radii_at_upper_bound(Δ::AbstractMatrix, Δ_max::Real) = all(v -> isnan(v) || v >= Δ_max, Δ)


"""
    update_trust_region!(algo::SCPAlgorithm, action::Symbol)

Apply a trust-region update `action ∈ (:shrink, :expand, :keep)` to `algo.tr` (and to
`algo.tr_u` when the control trust-region is used), clamped to `algo.Δ_bounds`.
`NaN` radii are left unchanged.
Returns `true` if the trust-region size actually changed.
"""
function update_trust_region!(algo::SCPAlgorithm, action::Symbol)
    @assert action in (:shrink, :expand, :keep) "Invalid trust-region action: $action"
    action == :keep && return false

    function _update(Δ)
        if action == :shrink
            return scale_trust_radii(Δ, algo.alphas[1], algo.Δ_bounds[1], true)
        else
            return scale_trust_radii(Δ, algo.alphas[2], algo.Δ_bounds[2], false)
        end
    end

    # `isequal` so that `NaN` slots (mixed problems) do not count as a change
    Δ_new = _update(algo.tr.Δ)
    changed = !isequal(Δ_new, algo.tr.Δ)
    algo.tr.Δ = Δ_new

    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        Δu_new = _update(algo.tr_u.Δ)
        changed = changed || !isequal(Δu_new, algo.tr_u.Δ)
        algo.tr_u.Δ = Δu_new
    end
    return changed
end


"""
    update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)

Update trust-region size Δ from the ratio `rho_i`: shrink if `rho_i < rhos[2]`, expand if
`rho_i >= rhos[3]`, keep otherwise. `NaN` radii are left unchanged.
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


"""
    trust_region_at_lower_bound(algo::SCPAlgorithm)

True when every trust-region radius (state, and control if used) is already at its lower bound
`Δ_bounds[1]`, so a rejected step cannot shrink the subproblem any further.
`NaN` radii (unused slots of mixed problems) are ignored.
"""
function trust_region_at_lower_bound(algo::SCPAlgorithm)
    at_bound = radii_at_lower_bound(algo.tr.Δ, algo.Δ_bounds[1])
    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        at_bound = at_bound && radii_at_lower_bound(algo.tr_u.Δ, algo.Δ_bounds[1])
    end
    return at_bound
end


"""
    clamp_trust_region!(algo::SCPAlgorithm)

Clamp the trust-region radii into `algo.Δ_bounds` (e.g. an initial `Δ0` below `Δ_min`, where the
subproblem solution is below the solver accuracy and the trust-region tests are meaningless).
`NaN` radii (unused slots of mixed problems) stay `NaN`.
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


"""Duals of one trust-region constraint entry: a vector constraint, an array of scalar constraints, or a scalar constraint"""
_trust_region_duals(c::AbstractArray) = [dual(ci) for ci in c]
_trust_region_duals(c) = vcat(dual(c))


"""
Accumulate `Σ |ν_lb + ν_ub| ⋅ (Δ_max - Δ)` over a pair of trust-region constraint containers.
Entry `k` covers rows `1:nrows` of column `k` of `Δ`; entry `(k, i)` (mixed problems) covers
`Δ[i, k]`. `NaN` radii (unused slots) are skipped.
"""
function _trust_region_dual_bound(cons_lb, cons_ub, Δ::AbstractMatrix, Δ_max::Real)
    bound = 0.0
    for key in eachindex(cons_lb)       # `keys` is not supported by JuMP's SparseAxisArray (mixed problems)
        idx = Tuple(key)
        ν = abs.(_trust_region_duals(cons_lb[idx...])) .+ abs.(_trust_region_duals(cons_ub[idx...]))
        if length(idx) == 1
            radii = view(Δ, 1:length(ν), idx[1])
        else
            radii = [Δ[idx[2], idx[1]]]
        end
        for (νi, Δi) in zip(ν, radii)
            isnan(Δi) && continue
            bound += νi * max(Δ_max - Δi, 0.0)
        end
    end
    return bound
end


"""
    trust_region_dual_bound(algo, prob)

Upper bound on how much more the convex model could decrease if every trust-region radius were
raised to its ceiling `Δ_bounds[2]`, computed from the multipliers `ν` of the trust-region
constraints of the subproblem that was just solved:

    L*(current Δ) - L*(Δ_max)  ≤  Σ ν ⋅ (Δ_max - Δ)

The bound follows from convexity: the optimal value of a convex program is a convex function of
the right-hand sides of its constraints, and `-ν` is a subgradient of it. A bound below the
optimality tolerance means the trust-region does not limit the step (to that tolerance) at any
radius the algorithm may use, even when the step lies on its boundary.

Returns `nothing` when the solver did not return duals or the constraints are not found.
"""
function trust_region_dual_bound(algo::SCPAlgorithm, prob::OptimalControlProblem)
    model = prob.model
    has_duals(model) || return nothing
    haskey(model, :constraint_trust_region_x_lb) || return nothing
    Δ_max = algo.Δ_bounds[2]
    bound = _trust_region_dual_bound(model[:constraint_trust_region_x_lb], model[:constraint_trust_region_x_ub],
                                     algo.tr.Δ, Δ_max)
    if hasproperty(algo, :use_trustregion_control) && algo.use_trustregion_control && !isnothing(algo.tr_u)
        haskey(model, :constraint_trust_region_u_lb) || return nothing
        bound += _trust_region_dual_bound(model[:constraint_trust_region_u_lb], model[:constraint_trust_region_u_ub],
                                          algo.tr_u.Δ, Δ_max)
    end
    return bound
end


"""
    trust_region_limiting(algo, prob, x, u, x_ref, u_ref; tol_dual = -1.0)

Whether the trust-region limits the subproblem step. The step must lie on the trust-region
boundary (`trust_region_active`), and, when `tol_dual > 0` and the solver returned duals, the
multipliers must also matter: `trust_region_dual_bound(algo, prob) > tol_dual`. A step on the
boundary whose multipliers are negligible (degenerate or weakly active trust-region) is then
treated like an unbounded step for the stationarity tests. `tol_dual ≤ 0` keeps the purely
geometric test.

Must be called before the model is modified, so that the duals of the last solve are available.
"""
function trust_region_limiting(algo::SCPAlgorithm, prob::OptimalControlProblem, x, u, x_ref, u_ref;
                               tol_dual::Real = -1.0)
    trust_region_active(algo, x, u, x_ref, u_ref) || return false
    tol_dual > 0 || return true
    bound = trust_region_dual_bound(algo, prob)
    isnothing(bound) && return true
    return bound > tol_dual
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
        # mixed problems hold a `NaN` constant (not a variable) in the unused continuous column `N`
        u[I] isa VariableRef || continue
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
                  tol_ΔL = 1e-8, tol_abs = Inf, tol_ΔL_null = 1e-3)

Classify a trust-region step from the actual (`ΔJ`) and predicted (`ΔL`) merit reductions.
Returns `(rho, kind)`, where `rho = ΔJ/ΔL` (or `NaN` if `ΔL == 0`) and `kind` is one of:

- `:restoration`: the reference does not satisfy the convex constraints (`ref_feasible = false`,
  or `ΔL < -ε` while the reference is still the initial guess, which indicates it: for a
  convex-feasible `z̄` and a consistent model, `L(z̄) = J(z̄)` and `ΔL ≥ 0`). The ratio is
  meaningless; the step is accepted without a ratio test.
- `:ratio`: `ΔL > ε`; the regular ratio test applies to `rho`.
- `:stationary`: `-ε_null ≤ ΔL ≤ ε` and the trust-region is inactive at the candidate (null
  step). The reference is feasible for its own subproblem with `L(z̄) = J(z̄)`, so the subproblem
  cannot improve on it by more than `ε` (a slightly negative `ΔL` only means the solver returned
  a point marginally worse than `z̄`, i.e. `z̄` is optimal to within the solver accuracy); since
  the trust-region is inactive, `z̄` is a stationary point of the penalized problem.
  The negative side is bounded by `ε_null = max(ε, tol_ΔL_null * max(1, |J_ref|), tol_abs)`
  (`tol_abs` only if it is a positive finite number). The solver noise in `ΔL` can exceed `ε`
  and the optimality tolerance: `z̄` satisfies the convex constraints only to the solver accuracy,
  and the penalty weight amplifies the correction the subproblem has to make (ΔL ≈ -3e-5 with
  J ≈ 0.2 has been observed both at w = 1e3 and at w ≈ 3e12). A candidate worse than `z̄` by more
  than `ε_null` is not noise but a failed solve or an inconsistent model.
- `:unresolved`: `|ΔL| ≤ ε` while the step is bounded by the trust-region: the predicted decrease
  is below resolution because the trust-region is too small to tell. The reference is kept and
  the trust-region is expanded (shrinking would only make `ΔL` smaller).
- `:reject`: `ΔL < -ε` while the step is bounded by the trust-region, or `ΔL < -ε_null` while it
  is not (inconsistent model or failed subproblem solve); the trust-region is shrunk.

The threshold is `ε = min(tol_ΔL * max(1, |J_ref|), tol_abs)`: relative to the merit to
account for solver accuracy, but never above `tol_abs` (the optimality tolerance), so that a
constant offset in the objective cannot hide a decrease the stopping test would care about.
A non-positive `tol_abs` (convergence test disabled) leaves `ε` uncapped.
"""
function classify_step(
    ΔJ::Real, ΔL::Real, J_ref::Real,
    ref_feasible::Bool, ref_is_initial_guess::Bool, tr_active::Bool;
    tol_ΔL::Real = 1e-8,
    tol_abs::Real = Inf,
    tol_ΔL_null::Real = 1e-3,
)
    ε = tol_ΔL * max(1.0, abs(J_ref))
    if tol_abs > 0          # a non-positive tol_abs (e.g. tol_opt = -1 to disable convergence) leaves ε uncapped
        ε = min(ε, tol_abs)
    end
    rho = ΔL == 0 ? NaN : ΔJ / ΔL
    if !ref_feasible || (ref_is_initial_guess && ΔL < -ε)
        return rho, :restoration
    elseif ΔL > ε
        return rho, :ratio
    elseif !tr_active
        # a slightly negative ΔL only means the solver returned a point marginally worse than z̄;
        # a candidate much worse than z̄ is not solver noise but a failed subproblem solve or an
        # inconsistent model, so it cannot certify stationarity
        ε_null = max(ε, tol_ΔL_null * max(1.0, abs(J_ref)))
        if 0 < tol_abs < Inf
            ε_null = max(ε_null, tol_abs)
        end
        return rho, (ΔL >= -ε_null ? :stationary : :reject)
    elseif ΔL >= -ε
        return rho, :unresolved
    else
        return rho, :reject
    end
end


"""Whether every trust-region radius (state, and control if used) is at its upper bound `Δ_bounds[2]`; `NaN` radii are ignored"""
function trust_region_at_ceiling(algo::SCPAlgorithm)
    at_ceiling = radii_at_upper_bound(algo.tr.Δ, algo.Δ_bounds[2])
    if algo.use_trustregion_control && !isnothing(algo.tr_u)
        at_ceiling = at_ceiling && radii_at_upper_bound(algo.tr_u.Δ, algo.Δ_bounds[2])
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
        add_control_trust_region!(algo, prob, u_ref)
    end
    return
end


function add_control_trust_region!(algo::SCPAlgorithm, prob::OptimalControlProblem, u_ref)
    @constraint(prob.model, constraint_trust_region_u_lb[k in 1:size(u_ref,2)],
        -(prob.model[:u][:,k] - u_ref[:,k]) <= algo.tr_u.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_u_ub[k in 1:size(u_ref,2)],
        prob.model[:u][:,k] - u_ref[:,k]  <= algo.tr_u.Δ[:,k])
    return
end
