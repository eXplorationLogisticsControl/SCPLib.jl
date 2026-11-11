"""SCvx algorithm"""


"""
SCvx algorithm

# Arguments
- `nx::Int`: number of states
- `N::Int`: number of time steps
- `Δ0`: initial (state) trust-region size
- `w::Float64`: exact (l1) penalty weight
- `rhos::Tuple{Real,Real,Real}`: trust-region acceptance thresholds
- `alphas::Tuple{Real,Real}`: trust-region size update factors
- `Δ_bounds::Tuple{Float64,Float64}`: trust-region size bounds
- `nu::Union{Nothing,Int}`: number of controls (optional; only used to build the control trust-region
  at construction, otherwise it is built from the problem at the first solve)
- `Δ0_u`: initial control trust-region size (used if `use_trustregion_control = true`)
- `use_trustregion_control::Bool`: whether to also impose a trust-region on the controls `u`

!!! note "Trust-region on the controls"
    By default (`use_trustregion_control = false`) the trust-region only bounds the states `x`.
    The convergence theory of SCvx (Mao et al. 2018) assumes a trust-region on all decision
    variables, which is what makes the linearization error (and hence the ratio test) vanish as
    Δ → 0. When the dynamics or the non-convex constraints are nonlinear in `u`, set
    `use_trustregion_control = true`.
"""
mutable struct SCvx <: TrustRegionAlgorithm
    # storage
    tr::TrustRegions
    tr_u::Union{Nothing,TrustRegions}

    # algorithm parameters
    w::Float64
    rhos::Tuple{Real,Real,Real}
    alphas::Tuple{Real,Real}
    Δ_bounds::Tuple{Float64,Float64}
    use_trustregion_control::Bool
    Δ0_u::Union{Float64,Vector{Float64},Matrix{Float64}}

    function SCvx(
        nx::Int,
        N::Int;
        Δ0::Union{Float64,Vector{Float64},Matrix{Float64}} = 0.05,
        w::Float64 = 1e3,
        rhos::Tuple{Real,Real,Real} = (0.0, 0.25, 0.7),
        alphas::Tuple{Real,Real} = (2.0, 3.0),
        Δ_bounds::Tuple{Float64,Float64} = (1e-6, 1e4),
        nu::Union{Nothing,Int} = nothing,
        Δ0_u::Union{Float64,Vector{Float64},Matrix{Float64}} = 0.1,
        use_trustregion_control::Bool = false,
    )
        tr = TrustRegions(nx, N, Δ0)
        # if `nu` is not provided, the control trust-region is built at the first solve
        tr_u = use_trustregion_control && !isnothing(nu) ? TrustRegions(nu, N, Δ0_u) : nothing

        new(tr, tr_u, w, rhos, alphas, Δ_bounds, use_trustregion_control, Δ0_u)
    end
end


function Base.show(io::IO, algo::SCvx)
    println(io, "SCvx algorithm")
    @printf("   Trust-region size Δ                                : %1.2e\n", algo.tr.Δ[1,1])
    @printf("   Penalty weight w                                   : %1.2e\n", algo.w)
    @printf("   Trust-region size bounds (Δ_min, Δ_max)            : %1.2e, %1.2e\n", algo.Δ_bounds[1], algo.Δ_bounds[2])
    @printf("   Use trust-region control for control variables     : %s\n", algo.use_trustregion_control ? "Yes" : "No")
end


"""
SCvx exact penalty function (l1-penalty)
"""
function penalty(algo::SCvx, prob::OptimalControlProblem, ξ_dyn, ξ, ζ)
    P = algo.w * norm(ξ_dyn,1)
    if prob.ng > 0
        P += algo.w * norm(ξ,1)
    end
    if prob.nh > 0
        P += algo.w * norm(ζ,1)
    end
    return P
end


"""
    add_l1_slacks!(algo::SCvx, prob::OptimalControlProblem)

Create the epigraph variables `t ≥ ‖ξ_dyn‖₁`, `t ≥ ‖ξ‖₁`, `t ≥ ‖ζ‖₁` of the exact penalty.
These do not depend on the reference, so they are created once per solve and reused at every
iteration (only the objective is rebuilt).
"""
function add_l1_slacks!(algo::SCvx, prob::OptimalControlProblem)
    slacks_L1 = Dict(
        :slack_gdyn => @variable(prob.model),
        :slack_gnoncvx => prob.ng > 0 ? @variable(prob.model) : nothing,
        :slack_hnoncvx => prob.nh > 0 ? @variable(prob.model) : nothing,
    )
    ξ_dyn = prob.model[:ξ_dyn]
    @constraint(prob.model, [slacks_L1[:slack_gdyn]; vec(ξ_dyn)] in MOI.NormOneCone(1 + length(ξ_dyn)))
    if prob.ng > 0
        ξ = prob.model[:ξ]
        @constraint(prob.model, [slacks_L1[:slack_gnoncvx]; ξ] in MOI.NormOneCone(1 + length(ξ)))
    end
    if prob.nh > 0
        # this penalization works because ζ is defined to be non-negative
        ζ = prob.model[:ζ]
        @constraint(prob.model, [slacks_L1[:slack_hnoncvx]; ζ] in MOI.NormOneCone(1 + length(ζ)))
    end
    return slacks_L1
end


"""
SCvx exact penalty function for the JuMP model, in terms of the epigraph variables created by
`add_l1_slacks!`
"""
function penalty(algo::SCvx, prob::OptimalControlProblem, ξ_dyn::Matrix{VariableRef}, ξ, ζ, slacks_L1)
    P = algo.w * slacks_L1[:slack_gdyn]
    if prob.ng > 0
        P += algo.w * slacks_L1[:slack_gnoncvx]
    end
    if prob.nh > 0
        P += algo.w * slacks_L1[:slack_hnoncvx]
    end
    return P
end


"""
Solve convex subproblem for SCvx algorithm

`slacks_L1` are the l1-penalty epigraph variables from `add_l1_slacks!`; when calling this
function repeatedly on the same problem, create them once and pass them in.
"""
function solve_convex_subproblem!(algo::SCvx, prob::OptimalControlProblem,
                                  slacks_L1::Dict = add_l1_slacks!(algo, prob))
    # set objective with penalty
    _ξ = prob.ng > 0 ? prob.model[:ξ] : nothing
    _ζ = prob.nh > 0 ? prob.model[:ζ] : nothing

    J = prob.objective(prob.model[:x], prob.model[:u])
    P = penalty(algo, prob, prob.model[:ξ_dyn], _ξ, _ζ, slacks_L1)
    @objective(prob.model, Min, J + P)

    # solve convex subproblem
    optimize!(prob.model)
    return
end


"""Solution struct for SCvx algorithm"""
mutable struct SCvxSolution <: SCPSolution
    status::Symbol
    x::Matrix
    u::Matrix
    n_iter::Int
    info::Dict

    function SCvxSolution(prob::OptimalControlProblem, Nu::Int)
        status = :Solving
        x = zeros(prob.nx, prob.N)
        u = initial_control_matrix(prob, Nu)
        
        info = Dict(
            :J0 => Float64[],
            :ΔJ => Float64[],
            :χ => Float64[],
            :Δ => Matrix{Float64}[],
            :accept => Bool[],
            :cpu_times => Dict(
                :time_subproblem => Float64[],
                :time_update_reference => Float64[],
                :time_iter_total => Float64[],
                :time_total => 0.0,
            )
        )
        new(status, x, u, 0, info)
    end
end


function Base.show(io::IO, solution::SCvxSolution)
    println(io, "SCvx solution")
    @printf("   Status                   : %s\n", solution.status)
    @printf("   Iterations               : %d\n", solution.n_iter)
    if length(solution.info[:J0]) > 0
        @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
    end
end


"""
Solve non-convex OCP with SCvx algorithm

# Keyword arguments
- `callback::Union{Nothing,Function}`: called each iteration as `callback(algo, solution, iteration, J0, χ)`
- `tol_ΔL::Float64`: relative tolerance on the predicted reduction; `|ΔL| ≤ ε` with
  `ε = min(tol_ΔL * max(1,|J_ref|), tol_opt)` is treated as zero (no model decrease); a
  negative `tol_opt` leaves `ε` uncapped and disables the stationarity-based exits
  (`:Optimal` from a null step, `:InfeasibleStationary`)
- `tol_ΔL_null::Float64`: relative bound on how negative `ΔL` may be in a null step;
  with the trust-region inactive, `ΔL < -ε_null`, `ε_null = max(ε, tol_ΔL_null * max(1,|J_ref|), tol_opt)`,
  is treated as a failed subproblem solve / inconsistent model and rejected instead of
  certifying the reference as stationary
- `tol_tr_dual::Float64`: tolerance of the dual test on the trust-region (default `tol_opt`). A step
  on the trust-region boundary still counts as not bounded by it when the trust-region multipliers
  bound the extra model decrease available up to `Δ_bounds[2]` by `tol_tr_dual`
  (see `trust_region_dual_bound`); `tol_tr_dual ≤ 0` keeps the purely geometric test
- `small_step_convergence::Bool`: also return `:Optimal` on an accepted step that is bounded by the
  trust-region when `|ΔJ| ≤ tol_opt`, `ΔL ≤ tol_opt` and `χ ≤ tol_feas`. This is a small-step
  test, not a stationarity certificate: it also passes when the trust-region has shrunk. Use it
  when the trust-region is deliberately kept small (e.g. a frozen radius on a time-dilation
  control), so it stays active; `solution.info[:converged_by]` is then `:small_step`

# Step classification and termination
Steps are classified as in `solve!(::SCvxStar, ...)` (restoration step from a convex-infeasible
initial guess, ratio test, null step, rejection); as there, the returned `solution` describes
the current iterate, so a rejected candidate is never returned. Termination statuses:
- `:Optimal`: an accepted step that is not bounded by the trust-region (see `tol_tr_dual`) has
  `|ΔJ| ≤ tol_opt` and `χ ≤ tol_feas` (`:ratio`); or the reference is a feasible stationary point
  of the subproblem (`:null`); or `J0 ≤ tol_J0` with `χ ≤ tol_feas` (`:tol_J0`); or, with
  `small_step_convergence = true`, the small-step test passes (`:small_step`). The test that
  fired is stored in `solution.info[:converged_by]`
- `:InfeasibleStationary`: the reference is a stationary point of the penalized problem
  (`ΔL ≈ 0`) but violates the non-convex constraints (Mao et al. 2018); increase `w`
- `:Stalled` / `:Feasible`: a step is rejected while the trust-region is already at its lower
  bound, so the subproblem cannot change; the last accepted reference is returned, with status
  `:Feasible` if it satisfies `χ ≤ tol_feas` and `:Stalled` otherwise
- `:CPFailed`, `:Feasible`, `:MaxIterReached`
"""
function solve!(
    algo::SCvx,
    prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint};
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-6,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
    callback::Union{Nothing,Function} = nothing,
    warmstart_primal::Bool = false,
    warmstart_dual::Bool = false,
    tol_ΔL::Float64 = 1e-8,
    tol_ΔL_null::Float64 = 1e-3,
    tol_tr_dual::Float64 = tol_opt,
    small_step_convergence::Bool = false,
)
    ensure_trust_region_control!(algo, prob)
    clamp_trust_region!(algo)
    if algo.use_trustregion_control
        @assert prob.nu == size(algo.tr_u.Δ,1) "Number of control variables mismatch between problem and algorithm"
    elseif verbosity > 0
        @warn("SCvx trust-region only bounds the states x; set `use_trustregion_control = true` " *
              "when the dynamics or constraints are nonlinear in u (see `?SCvx`)", maxlog = 1)
    end
    mask_unused_control_trust_region!(algo, prob)
    tcpu_start = time()

    # check whether the initial guess satisfies the convex constraints (before any linearized or
    # trust-region constraints are added); if not, the first step is a restoration step
    ref_feasible = reference_convex_feasible(prob, x_ref, u_ref; atol = max(tol_feas, sqrt(eps())))
    ref_is_initial_guess = true

    # l1-penalty epigraph variables, created once
    slacks_L1 = add_l1_slacks!(algo, prob)

    # initialize algorithm hyperparameters
    flag_reference    = true    # at initial iteraiton, we need to update reference
    flag_trust_region = true    # (redundant since `flag_reference = true`)
    δ_i = 1e16

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    # initialize solution struct
    solution = SCvxSolution(prob, size(u_ref,2))

    # print initial information
    header = "\nIter |     J0     |    ΔJ_i    |    ΔL_i    |     χ_i    |    ρ_i    |    r_i    |     w     |  acpt. |"
    if verbosity > 0
        println()
        @printf(" Solving OCP with SCvx Algorithm (`・ω・´)\n\n")
        @printf("   Feasibility tolerance tol_feas : % 1.2e\n", tol_feas)
        @printf("   Optimality tolerance tol_opt   : % 1.2e\n", tol_opt)
        @printf("   Objective tolerance tol_J0     : % 1.2e\n", tol_J0)
        @printf("   Penalty weight w               : % 1.2e\n", algo.w)
        @printf("   Warmstart primal               :  %s\n", warmstart_primal ? "Yes" : "No")
        @printf("   Warmstart dual                 :  %s\n", warmstart_dual ? "Yes" : "No")
        @printf("   Use trust-region on u          :  %s\n", algo.use_trustregion_control ? "Yes" : "No")
        println(header)
    end

    # initialize warmstart dictionaries
    variable_primal = Dict()
    constraint_solution = Dict()

    # append trust-region constraints references for nonlinear program
    if algo.use_trustregion_control
        for ref in (:constraint_trust_region_u_lb, :constraint_trust_region_u_ub)
            ref in prob.model_nl_references || push!(prob.model_nl_references, ref)
        end
    end


    for it in 1:maxiter
        tcpu_start_iter = time()
        # re-set non-convex expression according to reference
        if flag_reference
            if it > 1
                delete_noncvx_referencs!(prob, prob.model_nl_references)
            end
            g_dyn_ref, g_ref, h_ref = set_linearized_constraints!(prob, x_ref, u_ref)
            set_trust_region_constraints!(algo, prob, x_ref, u_ref)   # if ref is updated, we need to update trust region constraints
        
        # only update trust-region constraints
        elseif flag_trust_region
            if it > 1
                delete_noncvx_referencs!(prob, [:constraint_trust_region_x_lb, :constraint_trust_region_x_ub])
                if algo.use_trustregion_control
                    delete_noncvx_referencs!(prob, [:constraint_trust_region_u_lb, :constraint_trust_region_u_ub])
                end
            end
            set_trust_region_constraints!(algo, prob, x_ref, u_ref)   # if ref is not updated but trsut region size changed
        end
        push!(solution.info[:cpu_times][:time_update_reference], time() - tcpu_start_iter)

        # warmstart
        if it > 1 && (warmstart_primal || warmstart_dual)
            set_optimal_start_values(variable_primal, constraint_solution)
        end

        # solve convex subproblem
        tstart_cp = time()
        solve_convex_subproblem!(algo, prob, slacks_L1)
        push!(solution.info[:cpu_times][:time_subproblem], time() - tstart_cp)

        # check termination status
        if termination_status(prob.model) == SLOW_PROGRESS
            @warn("CP termination status: $(termination_status(prob.model))")
        elseif termination_status(prob.model) ∉ [OPTIMAL, ALMOST_OPTIMAL, LOCALLY_SOLVED]
            if verbosity > 0
                @warn("Exiting as CP termination status: $(termination_status(prob.model))")
            end
            solution.status = :CPFailed
            push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
            break
        end

        _x = value.(prob.model[:x])
        _u = control_values(prob)
        _ξ_dyn = value.(prob.model[:ξ_dyn])
        _ξ = prob.ng > 0 ? value.(prob.model[:ξ]) : nothing
        _ζ = prob.nh > 0 ? value.(prob.model[:ζ]) : nothing

        # evaluate nonlinear constraints
        _, g_dynamics = get_trajectory(prob, _x, _u)
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(prob.lincache, _x, _u) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(prob.lincache, _x, _u), 0) : nothing

        # merit of the reference, actual and predicted merit of the candidate
        J0_ref = prob.objective(x_ref, u_ref)
        J_ref = J0_ref + penalty(algo, prob, g_dyn_ref, g_ref, h_ref)
        J0 = prob.objective(_x, _u)
        J = J0 + penalty(algo, prob, g_dynamics, g_noncvx, h_noncvx)
        L = J0 + penalty(algo, prob, _ξ_dyn, _ξ, _ζ)
        ΔJ = J_ref - J            # actual cost reduction, eqn (13a)
        ΔL = J_ref - L            # predicted cost reduction, eqn (13b)

        # constraint violation of the candidate and of the reference
        χ = constraint_violation(prob, g_dynamics, g_noncvx, h_noncvx)
        χ_ref = constraint_violation(prob, g_dyn_ref, g_ref, h_ref)

        # classify step
        # the trust-region counts as active only if the step is on its boundary and (with duals)
        # its multipliers bound a model decrease above tol_tr_dual; read before the model changes
        tr_active = trust_region_limiting(algo, prob, _x, _u, x_ref, u_ref; tol_dual = tol_tr_dual)
        rho_i, step_kind = classify_step(ΔJ, ΔL, J_ref, ref_feasible, ref_is_initial_guess, tr_active;
                                         tol_ΔL = tol_ΔL, tol_abs = tol_opt, tol_ΔL_null = tol_ΔL_null)
        if step_kind == :unresolved && trust_region_at_ceiling(algo)
            step_kind = :stationary     # no resolvable decrease even at the largest trust-region
        end
        accept = step_kind == :restoration || (step_kind == :ratio && rho_i >= algo.rhos[1])
        tr_action = trust_region_action(algo, step_kind, rho_i)
        # rejected step while the trust-region cannot shrink any further
        collapsed = !accept && tr_action == :shrink && trust_region_at_lower_bound(algo)

        if verbosity > 0
            if mod(it, 20) == 0
                println(header)
            end
            @printf(" %3.0f | % 1.3e | % 1.3e | % 1.3e | % 1.3e | % 1.2e | % 1.2e | % 1.2e |  %s   |\n",
                    it, J0, ΔJ, ΔL, χ, rho_i, algo.tr.Δ[1,1], algo.w,
                    message_step(step_kind, accept))
        end

        # iterate on record (held by `solution`, stored in `info`, passed to the callback): the
        # candidate if it becomes the new reference (accepted step) or already meets the objective
        # target `tol_J0` while feasible; otherwise the reference, which remains the current iterate
        # (rejected, unresolved and null steps, trust-region collapse), so a rejected candidate is
        # never returned, e.g. when `maxiter` is reached on a rejected step
        if accept || ((J0 <= tol_J0) && (χ <= tol_feas))
            x_rec, u_rec, J0_rec, ΔJ_rec, χ_rec = _x, _u, J0, ΔJ, χ
        else
            x_rec, u_rec, J0_rec, ΔJ_rec, χ_rec = x_ref, u_ref, J0_ref, 0.0, χ_ref
        end

        # update current solution
        solution.x[:,:] = x_rec
        solution.u[:,:] = u_rec

        if store_iterates
            push!(solution.info[:J0], J0_rec)
            push!(solution.info[:ΔJ], ΔJ_rec)
            push!(solution.info[:χ], χ_rec)
            push!(solution.info[:Δ], algo.tr.Δ)
            push!(solution.info[:accept], accept)
            solution.n_iter += 1
        end

        if !isnothing(callback)
            callback(algo, solution, it, J0_rec, χ_rec)
        end

        # convergence check: |ΔJ| is only a stationarity measure for an accepted step that is not
        # bounded by the trust-region; a null step certifies stationarity of the reference
        converged_ratio = step_kind == :ratio && accept && !tr_active && (abs(ΔJ) <= tol_opt) && (χ <= tol_feas)
        converged_null  = step_kind == :stationary && (tol_opt >= 0) && (χ_ref <= tol_feas)   # tol_opt < 0 disables convergence
        reached_J0      = (J0_rec <= tol_J0) && (χ_rec <= tol_feas)
        # opt-in small-step test for a step bounded by the trust-region (not a stationarity certificate)
        converged_small = small_step_convergence && step_kind == :ratio && accept && tr_active &&
                          (abs(ΔJ) <= tol_opt) && (ΔL <= tol_opt) && (χ <= tol_feas)
        if converged_ratio || converged_null || reached_J0 || converged_small
            solution.status = :Optimal
            solution.info[:converged_by] = converged_ratio ? :ratio :
                                           converged_null  ? :null  :
                                           reached_J0      ? :tol_J0 : :small_step
            push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
            break
        end

        # stationary point of the exact penalty function that is infeasible: with a fixed
        # penalty weight no further progress is possible (Mao et al. 2018)
        if step_kind == :stationary && (tol_opt >= 0) && (χ_ref > tol_feas)
            solution.status = :InfeasibleStationary
            if verbosity > 0
                @warn("Exiting at an infeasible stationary point of the penalized problem (χ = $(χ_ref)); consider increasing w")
            end
            push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
            break
        end

        # check step acceptance
        if accept
            flag_reference = true
            ref_is_initial_guess = false
            ref_feasible = true         # subproblem solutions satisfy the convex constraints
            x_ref[:,:] = _x
            u_ref[:,:] = _u

            # update primal and dual solutions
            if warmstart_primal
                variable_primal = get_primal_variables(prob.model)
            end
            if warmstart_dual
                constraint_solution = get_constraint_solutions(prob.model)
            end
        else
            flag_reference = false
        end

        # update trust-region
        flag_trust_region = update_trust_region!(algo, tr_action)

        # a rejected step at the lower bound cannot change the subproblem: stop at the reference
        if collapsed
            solution.status = χ_rec <= tol_feas ? :Feasible : :Stalled
            if verbosity > 0
                @warn("Exiting as trust-region collapsed to its lower bound Δ_min = $(algo.Δ_bounds[1])")
            end
            push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
            break
        end

        push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)

        if verbosity >= 2
            # extra information when verbosity >= 2
            println()
            @printf("       CPU time on iteration        : %1.2f sec\n", solution.info[:cpu_times][:time_iter_total][end])
            @printf("       CPU time on subproblem       : %1.2f sec\n", solution.info[:cpu_times][:time_subproblem][end])
            @printf("       CPU time on update reference : %1.2f sec\n", solution.info[:cpu_times][:time_update_reference][end])
            println()
        end
        
        # handle termination status when maximum number of iterations is reached
        if it == maxiter && solution.status == :Solving
            if χ_rec <= tol_feas
                solution.status = :Feasible
            else
                solution.status = :MaxIterReached
            end
        end
    end
    tcpu_end = time()
    solution.info[:cpu_times][:time_total] = tcpu_end - tcpu_start

    # print exit results
    if verbosity > 0
        println()
        @printf("   Status                   : %s\n", solution.status)
        if haskey(solution.info, :converged_by)
            @printf("   Converged by             : %s\n", solution.info[:converged_by])
        end
        @printf("   Iterations               : %d\n", solution.n_iter)
        @printf("   Total CPU time           : %1.2f sec\n", tcpu_end - tcpu_start)
        if solution.n_iter > 0
            @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
            @printf("   Objective improvement ΔJ : %1.4e (tol: %1.4e)\n", solution.info[:ΔJ][end], tol_opt)
            @printf("   Max constraint violation : %1.4e (tol: %1.4e)\n", solution.info[:χ][end], tol_feas)
        end
        println()
    end
    return solution
end
