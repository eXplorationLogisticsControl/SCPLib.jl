"""Prox-linear algorithm"""


"""
Prox-linear algorithm

The weights are defined such that the penalized objective of the subproblems are:

```math
L(Z) + w_ep * ||G(Z) + ∇G(Z_ref)*(Z - Z_ref)||_1 + w_prox/2 * ||Z - Z_ref||_2^2
```

where `L(Z)` is the original objective function, `G(Z)` are the non-convex constraints.

# Arguments
- `w_ep::Float64`: exact penalty term weight
- `w_prox::Float64`: proximal term weight
- `proximal_u::Bool`: whether to enforce proximal constraint on u
"""
mutable struct ProxLinear <: SCPAlgorithm
    # hyperparameters
    w_ep::Float64           # exact penalty term weight
    w_prox::Float64         # proximal term weight
    proximal_u::Bool

    function ProxLinear(
        w_ep::Float64 = 1e2,
        w_prox::Float64 = 1e0;
        proximal_u::Bool = false,
    )
        new(
            w_ep, w_prox,
            proximal_u,
        )
    end
end


function Base.show(io::IO, algo::ProxLinear)
    println(io, "Prox-linear algorithm")
    @printf("   Exact penalty term weight w_ep   : %1.4e\n", algo.w_ep)
    @printf("   Proximal term weight w_prox      : %1.4e\n", algo.w_prox)
    @printf("   Enforce proximal constraint on u : %s\n", algo.proximal_u ? "Yes" : "No")
end


function solve_convex_subproblem!(
    algo::ProxLinear, prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint},
)

    # prepare slack for proximal term
    Δvars = [
        reshape(prob.model[:x] - x_ref, prod(size(prob.model[:x])));
    ]
    if algo.proximal_u
        append!(Δvars, reshape(prob.model[:u] - u_ref, prod(size(prob.model[:u]))))
    end
    ϵ_proximal = @variable(prob.model)
    @constraint(prob.model, [ϵ_proximal, Δvars...] in SecondOrderCone())

    # L1 penalty on dynamics constraints
    ϵ_noncvx_dyn = @variable(prob.model, [1:prob.nx,1:prob.N-1])
    @constraint(prob.model, ϵ_noncvx_dyn .>= 0.0)
    @constraint(prob.model, [k in 1:prob.N-1], prob.model[:ξ_dyn][:,k] .<=  ϵ_noncvx_dyn[:,k])
    @constraint(prob.model, [k in 1:prob.N-1], prob.model[:ξ_dyn][:,k] .>= -ϵ_noncvx_dyn[:,k])

    # L1 penalty on non-convex equality constraints
    if prob.ng > 0
        # slack for L1 norm of non-convex equality constraints violation
        ϵ_noncvx_g = @variable(prob.model, [1:prob.ng])
        @constraint(prob.model, ϵ_noncvx_g .>= 0.0)
        @constraint(prob.model, prob.model[:ξ] .<=  ϵ_noncvx_g)
        @constraint(prob.model, prob.model[:ξ] .>= -ϵ_noncvx_g)
    else
        ϵ_noncvx_g = 0.0
    end

    # max(0, h(x,u)) penalty on non-convex inequality constraints
    if prob.nh > 0
        slack_h_eval = prob.model[:ζ]
    else
        slack_h_eval = [0.0]
    end

    # combine into objective function
    J = prob.objective(prob.model[:x], prob.model[:u])    # original objective function
    @objective(prob.model, Min, J + algo.w_ep*(sum(ϵ_noncvx_dyn) + sum(ϵ_noncvx_g) + sum(slack_h_eval)) + algo.w_prox/2*ϵ_proximal^2)

    # solve convex subproblem
    optimize!(prob.model)
    if prob.nh > 0
        return sum(value.(ϵ_noncvx_dyn)) + sum(value.(ϵ_noncvx_g)), sum(value.(slack_h_eval)), value(ϵ_proximal)
    else
        return sum(value.(ϵ_noncvx_dyn)) + sum(value.(ϵ_noncvx_g)), 0.0, value(ϵ_proximal)
    end
end


"""
Solution struct for prox-linear algorithm
"""
mutable struct ProxLinearSolution <: SCPSolution
    status::Symbol
    x::Matrix
    u::Matrix
    n_iter::Int
    info::Dict

    function ProxLinearSolution(prob::OptimalControlProblem, Nu::Int)
        status = :Solving
        x = zeros(prob.nx, prob.N)
        u = zeros(prob.nu, Nu)
        
        info = Dict(
            :J0 => Float64[],
            :ΔJ => Float64[],
            :χ => Float64[],
            :w => Float64[],
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


function Base.show(io::IO, solution::ProxLinearSolution)
    println(io, "Prox-linear solution")
    @printf("   Status                   : %s\n", solution.status)
    @printf("   Iterations               : %d\n", solution.n_iter)
    if length(solution.info[:J0]) > 0
        @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
    end
end


"""
Solve non-convex OCP with prox-linear algorithm

# Arguments
- `algo::ProxLinear`: algorithm struct
- `prob::OptimalControlProblem`: problem struct
- `x_ref`: reference state history, size `nx`-by-`N`
- `u_ref`: reference control history, size `nu`-by-`N-1`
- `maxiter::Int`: maximum number of iterations
- `tol_feas::Float64`: feasibility tolerance
- `tol_opt::Float64`: optimality tolerance
- `tol_J0::Real`: objective tolerance
- `verbosity::Int`: verbosity level
- `store_iterates::Bool`: whether to store iterates
"""
function solve!(
    algo::ProxLinear,
    prob::OptimalControlProblem,
    x_ref,
    u_ref;
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
    callback::Union{Nothing,Function} = nothing,
)
    # initialize algorithm hyperparameters
    flag_reference    = true    # at initial iteraiton, we need to update reference
    J0_ref = 1e12
    tcpu_start = time()

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    # initialize solution object
    solution = ProxLinearSolution(prob, size(u_ref,2))

    # print initial information
    header = "\nIter |     J0     |  nrm(G,1)  |  nrm(H,1)  |  nrm(ΔZ,2) |    χ_i    |  acpt. |"
    if verbosity > 0
        println()
        @printf(" Solving OCP with prox-linear Algorithm (`・ω・´)\n\n")
        @printf("   Feasibility tolerance tol_feas : % 1.2e\n", tol_feas)
        @printf("   Optimality tolerance tol_opt   : % 1.2e\n", tol_opt)
        @printf("   Objective tolerance tol_J0     : % 1.2e\n", tol_J0)
        println(header)
    end

    # remove constraint_trust_region_x_lb and constraint_trust_region_x_ub
    filter!(e->e≠:constraint_trust_region_x_lb, prob.model_nl_references)
    filter!(e->e≠:constraint_trust_region_x_ub, prob.model_nl_references)

    for it in 1:maxiter
        tcpu_start_iter = time()
        # re-set non-convex expression according to reference
        if flag_reference
            if it > 1
                delete_noncvx_referencs!(prob, prob.model_nl_references)
            end
            g_dyn_ref, g_ref, h_ref = set_linearized_constraints!(prob, x_ref, u_ref)
        end
        push!(solution.info[:cpu_times][:time_update_reference], time() - tcpu_start_iter)

        # solve convex subproblem
        tstart_cp = time()
        _ϵ_noncvx_g, _ϵ_noncvx_h, _ϵ_proximal = solve_convex_subproblem!(algo, prob, x_ref, u_ref)
        push!(solution.info[:cpu_times][:time_subproblem], time() - tstart_cp)

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

        # evaluate objective
        J0 = prob.objective(_x, _u)
        ΔJ = J0 - J0_ref

        # evaluate nonlinear constraints
        _, g_dynamics = get_trajectory(prob, _x, _u)
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(_x, _u) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(_x, _u), 0) : nothing

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
            @printf(" %3.0f | % 1.3e | % 1.3e | % 1.3e | % 1.3e |% 1.3e |  %s   |\n",
                    it, J0, _ϵ_noncvx_g, _ϵ_noncvx_h, _ϵ_proximal, χ,
                    "yes") #message_accept_step(rho_i >= algo.rhos[1]))
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u

        # check for convergence
        if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
            solution.status = :Optimal
            break
        end

        if store_iterates
            push!(solution.info[:J0], J0)
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            solution.n_iter += 1
        end
        if !isnothing(callback)
            callback(solution)
        end

        # update reference solution
        flag_reference = true
        x_ref[:,:] = _x
        u_ref[:,:] = _u
        J0_ref = J0
        
        if it == maxiter
            if χ <= tol_feas
                solution.status = :Feasible
            else
                solution.status = :MaxIterReached
            end
        end
        push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
    end
    tcpu_end = time()
    solution.info[:cpu_times][:time_total] = tcpu_end - tcpu_start

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