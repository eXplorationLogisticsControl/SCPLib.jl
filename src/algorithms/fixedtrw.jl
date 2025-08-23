"""Fixed trust-region & weight SCP algorithm"""


mutable struct FixedTRWSCP <: SCPAlgorithm
    # algorithm hyperparameters
    Δ::Float64
    w::Float64

    function FixedTRWSCP(Δ::Float64, w::Float64)
        new(Δ, w)
    end
end


function Base.show(io::IO, algo::FixedTRWSCP)
    println(io, "Fixed trust-region SCP algorithm")
    @printf("   Trust-region size Δ                                : %1.2e\n", algo.Δ)
    @printf("   Penalty weight w                                   : %1.2e\n", algo.w)
end


"""
Augmented Lagrangian penalty function
"""
function penalty(algo::FixedTRWSCP, prob::OptimalControlProblem, ξ_dyn, ξ, ζ)
    P = algo.w/2 * dot(ξ_dyn,ξ_dyn)        # dynamics violation penalty
    if prob.ng > 0
        P += algo.w/2 * dot(ξ,ξ)           # append equality constraints terms
    end
    if prob.nh > 0
        P += algo.w/2 * dot(ζ,ζ)           # append inequality constraints terms
    end
    return P
end


function solve_convex_subproblem!(algo::FixedTRWSCP, prob::OptimalControlProblem)
    # set objective with penalty
    _y = prob.ny > 0 ? prob.model[:y] : nothing
    _ξ = prob.ng > 0 ? prob.model[:ξ] : nothing
    _ζ = prob.nh > 0 ? prob.model[:ζ] : nothing

    J = prob.objective(prob.model[:x], prob.model[:u], _y)
    P = penalty(algo, prob, prob.model[:ξ_dyn], _ξ, _ζ)
    @objective(prob.model, Min, J + P)

    # solve convex subproblem
    optimize!(prob.model)
    return
end


function set_trust_region_constraints!(algo::FixedTRWSCP, prob::OptimalControlProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.Δ * ones(prob.nx))
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.Δ * ones(prob.nx))
    return
end


mutable struct FixedTRWSCPSolution
    status::Symbol
    x::Matrix
    u::Matrix
    y::Union{Nothing,Matrix}
    n_iter::Int
    info::Dict

    function FixedTRWSCPSolution(prob::OptimalControlProblem, Nu::Int)
        status = :Solving
        x = zeros(prob.nx, prob.N)
        u = zeros(prob.nu, Nu)
        y = prob.ny > 0 ? zeros(prob.ny) : nothing

        info = Dict(
            :J0 => Float64[],
            :ΔJ => Float64[],
            :χ => Float64[],
            :w => Float64[],
            :Δ => Float64[],
            :accept => Bool[],
        )

        new(status, x, u, y, 0, info)
    end
end


function Base.show(io::IO, solution::FixedTRWSCPSolution)
    println(io, "Fixed trust-region SCP solution")
    @printf("   Status                   : %s\n", solution.status)
    @printf("   Iterations               : %d\n", solution.n_iter)
    @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
end


"""
Solve non-convex OCP with fixed trust-region-weight SCP Algorithm

This algorithm employs a fixed trust-region & accepts every convex subproblem step.
Artificial infeasibility is avoided by introducing slack variables for nonlinear constraints that are
penalized quadratically with a fixed weight.

# Arguments
- `algo::FixedTRWSCP`: algorithm struct
- `prob::OptimalControlProblem`: problem struct
- `x_ref`: reference state history, size `nx`-by-`N`
- `u_ref`: reference control history, size `nu`-by-`N-1`
- `y_ref`: reference other variables, size `ny`
- `maxiter::Int`: maximum number of iterations
- `tol_feas::Float64`: feasibility tolerance
- `tol_opt::Float64`: optimality tolerance
- `tol_J0::Real`: objective tolerance
- `verbosity::Int`: verbosity level
- `store_iterates::Bool`: whether to store iterates
"""
function solve!(
    algo::FixedTRWSCP,
    prob::OptimalControlProblem,
    x_ref, u_ref, y_ref;
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
)
    @assert tol_feas <= algo.w "tol_feas must be less than or equal to penalty weight w, currently set at $(algo.w)"
    tcpu_start = time()

    # print initial information
    if verbosity > 0
        println()
        @printf(" Solving OCP with Fixed Trust-Region & Weight SCP Algorithm (`・ω・´)\n\n")
        @printf("   Trust-region size Δ            : % 1.2e\n", algo.Δ)
        @printf("   Penalty weight w               : % 1.2e\n", algo.w)
        @printf("   Feasibility tolerance tol_feas : % 1.2e\n", tol_feas)
        @printf("   Optimality tolerance tol_opt   : % 1.2e\n", tol_opt)
        @printf("   Objective tolerance tol_J0     : % 1.2e\n", tol_J0)
        println()
    end

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    _y = y_ref isa Nothing ? nothing : similar(y_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing
    solution = FixedTRWSCPSolution(prob, size(u_ref,2))

    header = "\nIter |      J0      |    ΔJ_i     |    ΔL_i     |     χ_i     |     ρ_i     |"
    if verbosity > 0
        println(header)
    end
    cpu_times = Dict(
        :time_subproblem => 0.0,
        :time_update_reference => 0.0,
        :time_total => 0.0,
    )

    for it in 1:maxiter
        tcpu_start_iter = time()
        # re-set non-convex expression according to reference
        if it > 1
            delete_noncvx_referencs!(prob, prob.model_nl_references)
        end
        g_dyn_ref, g_ref, h_ref = set_linearized_constraints!(prob, x_ref, u_ref, y_ref)
        set_trust_region_constraints!(algo, prob, x_ref, u_ref)   # if ref is updated, we need to update trust region constraints
        cpu_times[:time_update_reference] = time() - tcpu_start_iter

        # solve convex subproblem
        tstart_cp = time()
        solve_convex_subproblem!(algo, prob)
        cpu_times[:time_subproblem] = time() - tstart_cp

        # check termination status
        if termination_status(prob.model) == SLOW_PROGRESS
            @warn("CP termination status: $(termination_status(prob.model))")
        elseif termination_status(prob.model) ∉ [OPTIMAL, ALMOST_OPTIMAL]
            if verbosity > 0
                @warn("Exiting as CP termination status: $(termination_status(prob.model))")
            end
            solution.status = :CPFailed
            break
        end

        _x = value.(prob.model[:x])
        _u = value.(prob.model[:u])
        _y = y_ref isa Nothing ? nothing : value.(prob.model[:y])
        _ξ_dyn = value.(prob.model[:ξ_dyn])
        _ξ = prob.ng > 0 ? value.(prob.model[:ξ]) : nothing
        _ζ = prob.nh > 0 ? value.(prob.model[:ζ]) : nothing

        # evaluate nonlinear constraints
        _, g_dynamics = get_trajectory(prob, _x, _u, _y)
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(_x, _u, _y) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(_x, _u, _y), 0) : nothing

        # check improvement
        J_ref = prob.objective(x_ref, u_ref, y_ref) + penalty(algo, prob, g_dyn_ref, g_ref, h_ref)
        J0 = prob.objective(_x, _u, _y)
        J = J0 + penalty(algo, prob, g_dynamics, g_noncvx, h_noncvx)
        L = J0 + penalty(algo, prob, _ξ_dyn, _ξ, _ζ)
        ΔJ = J_ref - J            # actual cost reduction, eqn (13a)
        ΔL = J_ref - L            # predicted cost reduction, eqn (13b)
        rho_i = abs(ΔL) > 1e-12 ? ΔJ / ΔL : 1.0

        # check nonlinear convergence
        χ = norm(g_dynamics,Inf)
        if prob.ng > 0
            χ = max(χ, norm(g_noncvx,Inf))
        end
        if prob.nh > 0
            χ = max(χ, norm(h_noncvx,Inf))
        end

        if verbosity > 0
            if mod(it, 20) == 0
                println(header)
            end
            @printf(" %3.0f | % 1.5e | % 1.4e | % 1.4e | % 1.4e | % 1.4e |\n",
                    it, J0, ΔJ, ΔL, χ, rho_i)
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u
        solution.y = _y

        # update reference
        x_ref[:,:] = _x
        u_ref[:,:] = _u
        if prob.ny > 0
            y_ref[:] = _y
        end

        if store_iterates
            push!(solution.info[:J0], J0)
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            solution.n_iter += 1
        end
        if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
            solution.status = :Optimal
            break
        end
        cpu_times[:time_total] = time() - tcpu_start_iter

        if verbosity >= 2
            # extra information when verbosity >= 2
            println()
            @printf("       CPU time on iteration        : %1.2f sec\n", cpu_times[:time_total])
            @printf("       CPU time on subproblem       : %1.2f sec\n", cpu_times[:time_subproblem])
            @printf("       CPU time on update reference : %1.2f sec\n", cpu_times[:time_update_reference])
            println()
        end

        if it == maxiter
            if χ <= tol_feas
                solution.status = :Feasible
            else
                solution.status = :MaxIterReached
            end
        end
    end
    tcpu_end = time()

    # print exit results
    if verbosity > 0
        println()
        @printf("   Status                   : %s\n", solution.status)
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
