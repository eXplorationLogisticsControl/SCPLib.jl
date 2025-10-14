"""Prox-linear algorithm"""


mutable struct ProxLinear <: SCPAlgorithm
    # hyperparameters
    w_ep::Float64           # non-convex linearization weight
    w_prox::Float64         # proximal term weight
    proximal_u::Bool

    function ProxLinear(
        w_ep::Float64,
        w_prox::Float64;
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
    @printf("   L1 penalization weight w_ep      : %1.4e\n", algo.w_ep)
    @printf("   Proximal term weight w_prox      : %1.4e\n", algo.w_prox)
    @printf("   Enforce proximal constraint on u : %s\n", algo.proximal_u ? "Yes" : "No")
end


"""
Solution struct for prox-linear algorithm
"""
mutable struct ProxLinearSolution <: SCPSolution
    status::Symbol
    x::Matrix
    u::Matrix
    y::Union{Nothing,Matrix}
    n_iter::Int
    info::Dict

    function ProxLinearSolution(prob::OptimalControlProblem, Nu::Int)
        status = :Solving
        x = zeros(prob.nx, prob.N)
        u = zeros(prob.nu, Nu)
        y = prob.ny > 0 ? zeros(prob.ny) : nothing
        
        info = Dict(
            :J0 => Float64[],
            :ΔJ => Float64[],
            :χ => Float64[],
            :w => Float64[],
            :Δ => Matrix{Float64}[],
            :accept => Bool[],
        )
        new(status, x, u, y, 0, info)
    end
end


function solve_convex_subproblem!(
    algo::ProxLinear, prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing}
)
    # define additional variables if they do not exist already
    _y = prob.ny > 0 ? prob.model[:y] : nothing

    # prepare slack for proximal term
    if algo.proximal_u
        Δvars = [
            reshape(prob.model[:x] - x_ref, prod(size(prob.model[:x])));
            reshape(prob.model[:u] - u_ref, prod(size(prob.model[:u])));
        ]
    else
        Δvars = [
            reshape(prob.model[:x] - x_ref, prod(size(prob.model[:x])));
        ]
    end
    ϵ_proximal = @variable(prob.model)
    @constraint(prob.model, [ϵ_proximal, Δvars...] in SecondOrderCone())

    # L1 penalty on non-convex constraints
    ng_dyn = prod(size(prob.model[:ξ_dyn]))
    ϵ_dynamics = @variable(prob.model)
    @constraint(prob.model, ϵ_dynamics >= 0)
    @constraint(prob.model,
        [ϵ_dynamics; vec(prob.model[:ξ_dyn])]
        in MOI.NormOneCone(1 + ng_dyn)
    )

    # combine into objective function
    J = prob.objective(prob.model[:x], prob.model[:u], _y)      # original objective function
    @objective(prob.model, Min, J + algo.w_ep*ϵ_dynamics + algo.w_prox/2*ϵ_proximal^2)

    # solve convex subproblem
    optimize!(prob.model)
    return value(ϵ_dynamics), value(ϵ_proximal)
end


function solve!(
    algo::ProxLinear,
    prob::OptimalControlProblem,
    x_ref, u_ref, y_ref;
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
)
    # initialize algorithm hyperparameters
    flag_reference    = true    # at initial iteraiton, we need to update reference
    J0_ref = 1e12
    tcpu_start = time()

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    _y = y_ref isa Nothing ? nothing : similar(y_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    # initialize solution object
    solution = ProxLinearSolution(prob, size(u_ref,2))

    # print initial information
    header = "\nIter |     J0     |  nrm(G,1)  |  nrm(ΔZ,2) |    χ_i    |  acpt. |"
    if verbosity > 0
        println()
        @printf(" Solving OCP with prox-linear Algorithm (`・ω・´)\n\n")
        @printf("   Feasibility tolerance tol_feas : % 1.2e\n", tol_feas)
        @printf("   Optimality tolerance tol_opt   : % 1.2e\n", tol_opt)
        @printf("   Objective tolerance tol_J0     : % 1.2e\n", tol_J0)
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
        if flag_reference
            if it > 1
                # delete_noncvx_referencs!(prob, prob.model_nl_references)
                delete_noncvx_referencs!(prob, [:constraint_dynamics,])
            end
            g_dyn_ref, g_ref, h_ref = set_linearized_constraints!(prob, x_ref, u_ref, y_ref)
        end
        cpu_times[:time_update_reference] = time() - tcpu_start_iter

        # solve convex subproblem
        tstart_cp = time()
        _ϵ_dynamics, _ϵ_proximal = solve_convex_subproblem!(algo, prob, x_ref, u_ref, y_ref)
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

        # evaluate objective
        J0 = prob.objective(_x, _u, _y)
        ΔJ = J0 - J0_ref

        # evaluate nonlinear constraints
        _, g_dynamics = get_trajectory(prob, _x, _u, _y)
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(_x, _u, _y) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(_x, _u, _y), 0) : nothing

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
            @printf(" %3.0f | % 1.3e | % 1.3e | % 1.3e |% 1.3e |  %s   |\n",
                    it, J0, _ϵ_dynamics, _ϵ_proximal, χ,
                    "yes") #message_accept_step(rho_i >= algo.rhos[1]))
        end

        # check for convergence
        if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
            solution.status = :Optimal
            break
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u
        solution.y = _y

        if store_iterates
            push!(solution.info[:J0], J0)
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            solution.n_iter += 1
        end

        # update reference solution
        flag_reference = true
        x_ref[:,:] = _x
        u_ref[:,:] = _u
        if prob.ny > 0
            y_ref[:] = _y
        end
        J0_ref = J0
        
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