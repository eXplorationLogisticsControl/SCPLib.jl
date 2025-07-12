"""SCvx* algorithm"""


"""
SCvx* algorithm
"""
mutable struct SCvxStar <: SCPAlgorithm
    # storage
    Δ::Float64
    w::Float64
    λ_dyn::Matrix
    λ::Vector
    μ::Vector

    # algorithm hyperparameters
    rhos::Tuple{Real,Real,Real}
    alphas::Tuple{Real,Real}
    Δ_bounds::Tuple{Float64,Float64}
    gamma::Float64
    beta::Float64
    w_max::Float64

    function SCvxStar(
        nx::Int,
        N::Int;
        ng::Int = 0,
        nh::Int = 0,
        Δ0::Float64 = 0.05,
        w0::Float64 = 1e2,
        rhos::Tuple{Real,Real,Real} = (0.0, 0.25, 0.7),
        alphas::Tuple{Real,Real} = (2.0, 3.0),
        Δ_bounds::Tuple{Float64,Float64} = (1e-6, 1e2),
        gamma::Float64 = 0.9,
        beta::Float64 = 2.0,
        w_max::Float64 = 1e16,
    )
        λ_dyn = zeros(nx, N-1)
        λ = zeros(ng)
        μ = zeros(nh)
        new(
            Δ0,
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
        )
    end
end


function penalty(algo::SCvxStar, prob::ContinuousProblem, ξ_dyn, ξ, ζ)
    # dynamics violation
    P = dot(algo.λ_dyn, ξ_dyn) + algo.w/2 * dot(ξ_dyn,ξ_dyn)

    # append equality constraints terms
    if prob.ng > 0
        P += dot(algo.λ, ξ) + algo.w/2 * dot(ξ,ξ)
    end

    # append inequality constraints terms
    if prob.nh > 0
        P += dot(algo.μ, ζ) + algo.w/2 * dot(ζ,ζ)
    end
    return P
end


function update_trust_region!(
    algo::SCvxStar,
    rho_i::Float64,
)
    flag_trust_region = false
    if rho_i < algo.rhos[2]
        algo.Δ = max(algo.Δ / algo.alphas[1], algo.Δ_bounds[1])
        flag_trust_region = true
    elseif rho_i >= algo.rhos[3]
        algo.Δ = min(algo.Δ * algo.alphas[2], algo.Δ_bounds[2])
        flag_trust_region = true
    end
    return flag_trust_region
end


function set_trust_region_constraints!(
    algo::SCvxStar,
    prob::ContinuousProblem,
    rho_i::Float64,
    x_ref,
    u_ref,
)
    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.Δ * ones(prob.nx))
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.Δ * ones(prob.nx))
    return
end


function solve_convex_subproblem!(
    algo::SCvxStar,
    prob::ContinuousProblem,
)
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


mutable struct SCvxStarSolution
    status::Symbol
    x::Matrix
    u::Matrix
    y::Union{Nothing,Matrix}
    n_iter::Int
    info::Dict

    function SCvxStarSolution(prob::ContinuousProblem)
        status = :Solving
        x = zeros(prob.nx, prob.N)
        u = zeros(prob.nu, prob.N-1)
        y = prob.ny > 0 ? zeros(prob.ny) : nothing
        
        info = Dict(
            :ΔJ => Float64[],
            :χ => Float64[],
            :w => Float64[],
            :Δ => Float64[],
            :accept => Bool[],
        )
        new(status, x, u, y, 0, info)
    end
end


function solve!(
    algo::SCvxStar,
    prob::ContinuousProblem,
    x_ref, u_ref, y_ref;
    maxiter::Int = 1,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
)
    # initialize
    rho_i = (algo.rhos[2] + algo.rhos[3]) / 2
    flag_reference    = true    # at initial iteraiton, we need to update reference
    flag_trust_region = true    # (redundant since `flag_reference = true`)
    δ_i = 1e16
    tcpu_start = time()

    _x = similar(x_ref)
    _u = similar(u_ref)
    _y = y_ref isa Nothing ? nothing : similar(y_ref)
    # _ξ_dyn = similar(prob.model[:ξ_dyn])
    # _ξ = prob.ng > 0 ? similar(prob.model[:ξ]) : nothing
    # _μ = prob.nh > 0 ? similar(prob.model[:μ]) : nothing 
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    solution = SCvxStarSolution(prob)

    header = "\nIter |     J0      |    ΔJ_i     |    ΔL_i     |     χ_i     |     ρ_i     |     r_i     |      w      |  step acpt. |"
    if verbosity > 0
        println(header)
    end

    for it in 1:maxiter
        # re-set non-convex expression according to reference
        if flag_reference
            if it > 1
                delete_noncvx_referencs!(prob, prob.model_nl_references)
            end
            g_dyn_ref, g_ref, h_ref = set_noncvx_expressions!(prob, x_ref, u_ref, y_ref)
            set_trust_region_constraints!(algo, prob, rho_i, x_ref, u_ref)   # if ref is updated, we need to update trust region constraints
        elseif flag_trust_region
            if it > 1
                delete_noncvx_referencs!(prob, [:constraint_trust_region_x_lb, :constraint_trust_region_x_ub])
            end
            set_trust_region_constraints!(algo, prob, rho_i, x_ref, u_ref)   # if ref is not updated but trsut region size changed
        end

        # solve convex subproblem
        solve_convex_subproblem!(algo, prob)

        # check termination status
        if termination_status(prob.model) == SLOW_PROGRESS
            @warn("CP termination status: $(termination_status(prob.model))")
        elseif termination_status(prob.model) ∉ [OPTIMAL, ALMOST_OPTIMAL]
            #!= OPTIMAL || (termination_status(prob.model) != ALMOST_OPTIMAL)
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
        g_noncvx = prob.ng > 0 ? evaluate_g_noncvx(prob, _x, _u, _y) : nothing
        h_noncvx = prob.nh > 0 ? evaluate_h_noncvx(prob, _x, _u, _y) : nothing

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
            @printf(" %3.0f | % 1.4e | % 1.4e | % 1.4e | % 1.4e | % 1.4e | % 1.4e | % 1.4e |     %s     |\n",
                    it, J0, ΔJ, ΔL, χ, rho_i, algo.Δ, algo.w,
                    message_accept_step(rho_i >= algo.rhos[1]))
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u
        solution.y = _y

        if store_iterates
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            push!(solution.info[:w], algo.w)
            push!(solution.info[:Δ], algo.Δ)
            push!(solution.info[:accept], rho_i >= algo.rhos[1])
            solution.n_iter += 1
        end

        if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
            solution.status = :Optimal
            break
        end
        
        # check step acceptance
        if rho_i >= algo.rhos[1]
            flag_reference = true
            x_ref[:,:] = _x
            u_ref[:,:] = _u
            if prob.ny > 0
                y_ref[:] = _y
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

        if it == maxiter
            solution.status = :MaxIterReached
        end
    end
    tcpu_end = time()

    # print exit results
    if verbosity > 0
        println()
        @printf("   Status                   : %s\n", solution.status)
        @printf("   Iterations               : %d\n", solution.n_iter)
        @printf("   Total CPU time           : %1.2f sec\n", tcpu_end - tcpu_start)
        @printf("   Objective improvement ΔJ : %1.4e (tol: %1.4e)\n", solution.info[:ΔJ][end], tol_opt)
        @printf("   Max constraint violation : %1.4e (tol: %1.4e)\n", solution.info[:χ][end], tol_feas)
        println()
    end
    return solution
end
