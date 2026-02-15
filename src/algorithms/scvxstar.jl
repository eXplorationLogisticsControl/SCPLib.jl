"""SCvx* algorithm"""


"""
SCvx* algorithm

# Arguments
- `nx::Int`: number of states
- `N::Int`: number of time steps
- `ng::Int`: number of non-convex equality constraints
- `nh::Int`: number of non-convex inequality constraints
- `Δ0::Float64`: initial trust-region size
- `w0::Float64`: initial penalty weight
- `rhos::Tuple{Real,Real,Real}`: trust-region acceptance thresholds
- `alphas::Tuple{Real,Real}`: trust-region size update factors
- `Δ_bounds::Tuple{Float64,Float64}`: trust-region size bounds
- `gamma::Float64`: stationarity tolerance update factor
- `beta::Float64`: penalty weight update factor
- `w_max::Float64`: maximum penalty weight
"""
mutable struct SCvxStar <: TrustRegionAlgorithm
    # storage
    tr::TrustRegions
    w::Union{Nothing,Float64}
    λ_dyn::Matrix
    λ::Vector
    μ::Vector

    # algorithm parameters
    rhos::Tuple{Real,Real,Real}
    alphas::Tuple{Real,Real}
    Δ_bounds::Tuple{Float64,Float64}
    gamma::Float64
    beta::Float64
    w_max::Float64
    l1_penalty::Bool

    function SCvxStar(
        nx::Int,
        N::Int;
        ng::Int = 0,
        nh::Int = 0,
        Δ0::Union{Float64,Vector{Float64},Matrix{Float64}} = 0.05,
        w0::Union{Nothing,Float64} = nothing,
        rhos::Tuple{Real,Real,Real} = (0.0, 0.25, 0.7),
        alphas::Tuple{Real,Real} = (2.0, 3.0),
        Δ_bounds::Tuple{Float64,Float64} = (1e-6, 1e4),
        gamma::Float64 = 0.9,
        beta::Float64 = 2.0,
        w_max::Float64 = 1e16,
        l1_penalty::Bool = false,
    )
        λ_dyn = zeros(nx, N-1)
        λ = zeros(ng)
        μ = zeros(nh)
        tr = TrustRegions(nx, N, Δ0)
        new(
            tr,
            w0,
            λ_dyn,
            λ,
            μ,  
            rhos,
            alphas,
            Δ_bounds,
            gamma,
            beta,
            w_max,
            l1_penalty,
        )
    end
end


function Base.show(io::IO, algo::SCvxStar)
    println(io, "SCvx* algorithm")
    @printf("   Trust-region size Δ                                : %1.2e\n", algo.tr.Δ[1,1])
    @printf("   Penalty weight w                                   : %1.2e\n", algo.w)
    @printf("   Penalty weight update factor β                     : %1.2e\n", algo.beta)
    @printf("   Use L1 penalty                                     : %s\n", algo.l1_penalty ? "Yes" : "No")
    @printf("   Maximum penalty weight w_max                       : %1.2e\n", algo.w_max)
    @printf("   Trust-region acceptance thresholds (ρ_1, ρ_2, ρ_3) : %1.2e, %1.2e, %1.2e\n", algo.rhos[1], algo.rhos[2], algo.rhos[3])
    @printf("   Trust-region size update factors (α_1, α_2)        : %1.2e, %1.2e\n", algo.alphas[1], algo.alphas[2])
end


"""
Augmented Lagrangian penalty function
"""
function penalty(algo::SCvxStar, prob::OptimalControlProblem, ξ_dyn::Matrix{Float64}, ξ, ζ)
    P = dot(algo.λ_dyn, ξ_dyn) + algo.w/2 * dot(ξ_dyn,ξ_dyn)        # dynamics violation penalty
    if prob.ng > 0
        P += dot(algo.λ, ξ) + algo.w/2 * dot(ξ,ξ)                   # append equality constraints terms
    end
    if prob.nh > 0
        P += dot(algo.μ, ζ) + algo.w/2 * dot(ζ,ζ)                   # append inequality constraints terms
    end

    if algo.l1_penalty
        P += sqrt(algo.w) * norm(ξ_dyn,1)
        if prob.ng > 0
            P += sqrt(algo.w) * norm(ξ,1)
        end
        if prob.nh > 0
            P += sqrt(algo.w) * norm(ζ,1)
        end
    end
    return P
end


function penalty(algo::SCvxStar, prob::OptimalControlProblem, ξ_dyn::Matrix{VariableRef}, ξ, ζ, slacks_L1)
    P = dot(algo.λ_dyn, ξ_dyn) + algo.w/2 * dot(ξ_dyn,ξ_dyn)        # dynamics violation penalty
    if prob.ng > 0
        P += dot(algo.λ, ξ) + algo.w/2 * dot(ξ,ξ)                   # append equality constraints terms
    end
    if prob.nh > 0
        P += dot(algo.μ, ζ) + algo.w/2 * dot(ζ,ζ)                   # append inequality constraints terms
    end

    if algo.l1_penalty
        P += sqrt(algo.w) * sum(slacks_L1[:slack_gdyn])
        @constraint(prob.model, [slacks_L1[:slack_gdyn]; vec(ξ_dyn)] in MOI.NormOneCone(1 + prod(size(ξ_dyn))))
        if prob.ng > 0
            P += sqrt(algo.w) * sum(slacks_L1[:slack_gnoncvx])
            @constraint(prob.model, [slacks_L1[:slack_gnoncvx]; ξ] in MOI.NormOneCone(1 + length(ξ)))
        end
        if prob.nh > 0
            # this penalization works because ζ is defined to be non-negative
            P += sqrt(algo.w) * sum(slacks_L1[:slack_hnoncvx])
            @constraint(prob.model, [slacks_L1[:slack_hnoncvx]; ζ] in MOI.NormOneCone(1 + length(ζ)))
        end
    end
    return P
end


# """Update trust-region size"""
# function update_trust_region!(algo::SCvxStar, rho_i::Float64)
#     flag_trust_region = false
#     if rho_i < algo.rhos[2]
#         algo.tr.Δ = max.(algo.tr.Δ / algo.alphas[1], algo.Δ_bounds[1])
#         flag_trust_region = true
#     elseif rho_i >= algo.rhos[3]
#         algo.tr.Δ = min.(algo.tr.Δ * algo.alphas[2], algo.Δ_bounds[2])
#         flag_trust_region = true
#     end
#     return flag_trust_region
# end


# """Set trust-region constraints"""
# function set_trust_region_constraints!(algo::SCvxStar, prob::OptimalControlProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
#     # define trust-region constraints
#     @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
#         -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.tr.Δ[:,k])
#     @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
#           prob.model[:x][:,k] - x_ref[:,k]  <= algo.tr.Δ[:,k])
#     return
# end


"""Solve convex subproblem for SCvx* algorithm"""
function solve_convex_subproblem!(algo::SCvxStar, prob::OptimalControlProblem)
    # set objective with penalty
    _ξ = prob.ng > 0 ? prob.model[:ξ] : nothing
    _ζ = prob.nh > 0 ? prob.model[:ζ] : nothing

    # append slack variable for l1 penalty term
    if algo.l1_penalty
        slacks_L1 = Dict(
            :slack_gdyn => @variable(prob.model),
            :slack_gnoncvx => prob.ng > 0 ? @variable(prob.model) : nothing,
            :slack_hnoncvx => prob.nh > 0 ? @variable(prob.model) : nothing,
        )
        #@variable(prob.model, [1:1+prob.ng+prob.nh])  # anonymous construction of variable
    else
        slacks_L1 = nothing
    end

    J = prob.objective(prob.model[:x], prob.model[:u])
    P = penalty(algo, prob, prob.model[:ξ_dyn], _ξ, _ζ, slacks_L1)
    @objective(prob.model, Min, J + P)

    # solve convex subproblem
    optimize!(prob.model)
    return
end


"""Solution struct for SCvx* algorithm"""
mutable struct SCvxStarSolution <: SCPSolution
    status::Symbol
    x::Matrix
    u::Matrix
    n_iter::Int
    info::Dict

    function SCvxStarSolution(prob::OptimalControlProblem, Nu::Int)
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


function Base.show(io::IO, solution::SCvxStarSolution)
    println(io, "SCvx* solution")
    @printf("   Status                   : %s\n", solution.status)
    @printf("   Iterations               : %d\n", solution.n_iter)
    if length(solution.info[:J0]) > 0
        @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
    end
end


function tune_initial_penalty_weight!(algo::SCvxStar, prob::OptimalControlProblem, x_ref, u_ref, J_expected::Real = 1.0, K_w::Real = 10.0)
    # evaluate nonlinear constraints
    if isnothing(prob.fun_get_trajectory)
        _, g_dynamics = get_trajectory(prob, x_ref, u_ref)
    else
        _, g_dynamics = prob.fun_get_trajectory(prob, x_ref, u_ref)
    end
    g_noncvx = prob.ng > 0 ? prob.g_noncvx(x_ref, u_ref) : nothing
    h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(x_ref, u_ref), 0) : nothing
    χ = norm(g_dynamics,Inf)
    if prob.ng > 0
        χ = max(χ, norm(g_noncvx,Inf))
    end
    if prob.nh > 0
        χ = max(χ, norm(h_noncvx,Inf))
    end

    # return initial penalty weight
    eps = 1e-8
    algo.w = K_w * (eps + J_expected) / (eps + χ)
    return
end


"""
Solve non-convex OCP with SCvx* algorithm

# Arguments
- `algo::SCvxStar`: algorithm struct
- `prob::OptimalControlProblem`: problem struct
- `x_ref`: reference state history, size `nx`-by-`N`
- `u_ref`: reference control history, size `nu`-by-`N-1`
- `maxiter::Int`: maximum number of iterations
- `tol_feas::Float64`: feasibility tolerance
- `tol_opt::Float64`: optimality tolerance
- `tol_J0::Real`: objective tolerance
- `J_expected::Real`: expected objective value
- `K_w::Real`: initial penalty weight scaling factor
- `verbosity::Int`: verbosity level
- `store_iterates::Bool`: whether to store iterates
- `callback::Union{Nothing,Function}`: callback function
- `warmstart_primal::Bool`: whether to warmstart primal variables
- `warmstart_dual::Bool`: whether to warmstart dual variables
"""
function solve!(
    algo::SCvxStar,
    prob::OptimalControlProblem,
    x_ref,
    u_ref;
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    J_expected::Real = 1.0,
    K_w::Real = 10.0,
    verbosity::Int = 1,
    store_iterates::Bool = true,
    callback::Union{Nothing,Function} = nothing,
    warmstart_primal::Bool = false,
    warmstart_dual::Bool = false,
)
    @assert prob.ng == length(algo.λ) "Number of non-convex equality constraints mismatch between problem and algorithm"
    @assert prob.nh == length(algo.μ) "Number of non-convex inequality constraints mismatch between problem and algorithm"
    tcpu_start = time()

    # re-tune initial penalty weight if not provided
    if isnothing(algo.w)
        tune_initial_penalty_weight!(algo, prob, x_ref, u_ref, J_expected, K_w)
    end

    # initialize algorithm hyperparameters
    rho_i = (algo.rhos[2] + algo.rhos[3]) / 2
    flag_reference    = true    # at initial iteraiton, we need to update reference
    flag_trust_region = true    # (redundant since `flag_reference = true`)
    δ_i = 1e16

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    # initialize solution object
    solution = SCvxStarSolution(prob, size(u_ref,2))

    # print initial information
    header = "\nIter |     J0     |    ΔJ_i    |    ΔL_i    |     χ_i    |    ρ_i    |    r_i    |     w     |  acpt. |"
    if verbosity > 0
        println()
        @printf(" Solving OCP with SCvx* Algorithm (`・ω・´)\n\n")
        @printf("   Feasibility tolerance tol_feas : % 1.2e\n", tol_feas)
        @printf("   Optimality tolerance tol_opt   : % 1.2e\n", tol_opt)
        @printf("   Objective tolerance tol_J0     : % 1.2e\n", tol_J0)
        @printf("   Initial penalty weight w       : % 1.2e\n", algo.w)
        @printf("   Use L1 penalty                 :  %s\n", algo.l1_penalty ? "Yes" : "No")
        @printf("   Warmstart primal               :  %s\n", warmstart_primal ? "Yes" : "No")
        @printf("   Warmstart dual                 :  %s\n", warmstart_dual ? "Yes" : "No")
        println(header)
    end

    # initialize warmstart dictionaries
    variable_primal = Dict()
    constraint_solution = Dict()

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
        solve_convex_subproblem!(algo, prob)
        push!(solution.info[:cpu_times][:time_subproblem], time() - tstart_cp)

        # check termination status
        if termination_status(prob.model) == SLOW_PROGRESS
            @warn("CP termination status: $(termination_status(prob.model))")
        elseif termination_status(prob.model) ∉ [OPTIMAL, ALMOST_OPTIMAL, LOCALLY_SOLVED]
            if verbosity > 0
                @warn("Exiting as CP termination status: $(termination_status(prob.model))")
            end
            solution.status = :CPFailed
            break
        end

        _x = value.(prob.model[:x])
        _u = value.(prob.model[:u])
        _ξ_dyn = value.(prob.model[:ξ_dyn])
        _ξ = prob.ng > 0 ? value.(prob.model[:ξ]) : nothing
        _ζ = prob.nh > 0 ? value.(prob.model[:ζ]) : nothing

        # evaluate nonlinear constraints
        if isnothing(prob.fun_get_trajectory)
            _, g_dynamics = get_trajectory(prob, _x, _u)
        else
            _, g_dynamics = prob.fun_get_trajectory(prob, _x, _u)
        end
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(_x, _u) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(_x, _u), 0) : nothing

        # check improvement
        J_ref = prob.objective(x_ref, u_ref) + penalty(algo, prob, g_dyn_ref, g_ref, h_ref)
        J0 = prob.objective(_x, _u)
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
            @printf(" %3.0f | % 1.3e | % 1.3e | % 1.3e | % 1.3e | % 1.2e | % 1.2e | % 1.2e |  %s   |\n",
                    it, J0, ΔJ, ΔL, χ, rho_i, algo.tr.Δ[1,1], algo.w,
                    message_accept_step(rho_i >= algo.rhos[1]))
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u

        if store_iterates
            push!(solution.info[:J0], J0)
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            push!(solution.info[:w], algo.w)
            push!(solution.info[:Δ], algo.tr.Δ)
            push!(solution.info[:accept], rho_i >= algo.rhos[1])
            solution.n_iter += 1
        end

        if !isnothing(callback)
            callback(solution)
        end

        if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
            solution.status = :Optimal
            push!(solution.info[:cpu_times][:time_iter_total], time() - tcpu_start_iter)
            break
        end
               
        # check step acceptance
        if rho_i >= algo.rhos[1]
            flag_reference = true
            x_ref[:,:] = _x
            u_ref[:,:] = _u

            # update primal and dual solutions
            if warmstart_primal
                variable_primal = get_primal_variables(prob.model)
            end
            if warmstart_dual
                constraint_solution = get_constraint_solutions(prob.model)
            end

            # stationarity check
            if abs(ΔJ) < δ_i
                algo.λ_dyn = algo.λ_dyn + algo.w * g_dynamics
                if prob.ng > 0
                    algo.λ = algo.λ + algo.w * g_noncvx
                end
                if prob.nh > 0
                    algo.μ = algo.μ + algo.w * h_noncvx
                end
                algo.w = min(algo.beta * algo.w, algo.w_max)

                # stationarity tolerance δ_i update - Ref. [1] Algorithm 1 line 15
                δ_i = δ_i > 1e12 ? abs(ΔJ) : δ_i * algo.gamma
            end
        else
            flag_reference = false
        end

        # update trust-region 
        flag_trust_region = update_trust_region!(algo, rho_i)

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
