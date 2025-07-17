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


function Base.show(io::IO, algo::SCvxStar)
    println(io, "SCvx* algorithm")
    @printf("   Trust-region size Δ                                : %1.2e\n", algo.Δ)
    @printf("   Penalty weight w                                   : %1.2e\n", algo.w)
    @printf("   Penalty weight update factor β                     : %1.2e\n", algo.beta)
    @printf("   Maximum penalty weight w_max                       : %1.2e\n", algo.w_max)
    @printf("   Trust-region acceptance thresholds (ρ_1, ρ_2, ρ_3) : %1.2e, %1.2e, %1.2e\n", algo.rhos[1], algo.rhos[2], algo.rhos[3])
    @printf("   Trust-region size update factors (α_1, α_2)        : %1.2e, %1.2e\n", algo.alphas[1], algo.alphas[2])
end


"""
Augmented Lagrangian penalty function
"""
function penalty(algo::SCvxStar, prob::OptimalControlProblem, ξ_dyn, ξ, ζ)
    P = dot(algo.λ_dyn, ξ_dyn) + algo.w/2 * dot(ξ_dyn,ξ_dyn)        # dynamics violation penalty
    if prob.ng > 0
        P += dot(algo.λ, ξ) + algo.w/2 * dot(ξ,ξ)                   # append equality constraints terms
    end
    if prob.nh > 0
        P += dot(algo.μ, ζ) + algo.w/2 * dot(ζ,ζ)                   # append inequality constraints terms
    end
    return P
end


"""Update trust-region size"""
function update_trust_region!(algo::SCvxStar, rho_i::Float64)
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


function set_trust_region_constraints!(algo::SCvxStar, prob::OptimalControlProblem, x_ref, u_ref)
    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.Δ * ones(prob.nx))
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.Δ * ones(prob.nx))
    return
end


function solve_convex_subproblem!(algo::SCvxStar, prob::OptimalControlProblem)
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


"""
Solution struct for SCvx* algorithm
"""
mutable struct SCvxStarSolution
    status::Symbol
    x::Matrix
    u::Matrix
    y::Union{Nothing,Matrix}
    n_iter::Int
    info::Dict

    function SCvxStarSolution(prob::OptimalControlProblem, Nu::Int)
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


function set_linearized_dynamics_constraints!(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref, y_ref)
    set_continuous_dynamics_cache!(prob.lincache, x_ref, u_ref, sols)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )
    return g_dynamics_ref
end


function set_linearized_dynamics_constraints!(prob::ImpulsiveProblem, x_ref, u_ref, y_ref)
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref, y_ref)
    set_impulsive_dynamics_cache!(prob.lincache, x_ref, u_ref, sols, prob.dfdu)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )
    return g_dynamics_ref
end


"""
Set linearized non-convex constraints for scvx* algorithm
"""
function set_linearized_constraints!(prob::OptimalControlProblem, x_ref, u_ref, y_ref)
    # set dynamics constraints
    g_dynamics_ref = set_linearized_dynamics_constraints!(prob, x_ref, u_ref, y_ref)

    # define stacked flattened variables difference
    if prob.ng > 0 || prob.nh > 0
        _Δy = y_ref isa Nothing ? nothing : prob.model[:y] - y_ref
        Δz = stack_flatten_variables(prob, prob.model[:x] - x_ref, prob.model[:u] - u_ref, _Δy)
    end

    # set nonconvex equality constraints
    if prob.ng > 0
        set_g_noncvx_cache!(prob.lincache, prob.∇g_noncvx, x_ref, u_ref, y_ref)
        g_ref = prob.g_noncvx(x_ref, u_ref, y_ref)
        @constraint(prob.model, constraint_g_noncvx[i in 1:prob.ng],
            g_ref[i] + prob.lincache.∇g[i,:]' * Δz == prob.model[:ξ][i]
        )
    else
        g_ref = nothing
    end

    # set nonconvex inequality constraints
    if prob.nh > 0
        set_h_noncvx_cache!(prob.lincache, prob.∇h_noncvx, x_ref, u_ref, y_ref)
        h_ref = max.(prob.h_noncvx(x_ref, u_ref, y_ref), 0)
        @constraint(prob.model, constraint_h_noncvx[i in 1:prob.nh],
            h_ref[i] + prob.lincache.∇h[i,:]' * Δz <= prob.model[:ζ][i]
        )
    else
        h_ref = nothing
    end
    return g_dynamics_ref, g_ref, h_ref
end



"""
Solve non-convex OCP with SCvx* algorithm

# Arguments
- `algo::SCvxStar`: algorithm struct
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
    algo::SCvxStar,
    prob::OptimalControlProblem,
    x_ref, u_ref, y_ref;
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-4,
    tol_J0::Real = -1e16,
    verbosity::Int = 1,
    store_iterates::Bool = true,
)
    @assert prob.ng == length(algo.λ) "Number of non-convex equality constraints mismatch between problem and algorithm"
    @assert prob.nh == length(algo.μ) "Number of non-convex inequality constraints mismatch between problem and algorithm"
    
    # initialize algorithm hyperparameters
    rho_i = (algo.rhos[2] + algo.rhos[3]) / 2
    flag_reference    = true    # at initial iteraiton, we need to update reference
    flag_trust_region = true    # (redundant since `flag_reference = true`)
    δ_i = 1e16
    tcpu_start = time()

    # initialize storage
    _x = similar(x_ref)
    _u = similar(u_ref)
    _y = y_ref isa Nothing ? nothing : similar(y_ref)
    g_dyn_ref = zeros(prob.nx,prob.N-1)
    g_ref = prob.ng > 0 ? zeros(prob.ng) : nothing
    h_ref = prob.nh > 0 ? zeros(prob.nh) : nothing

    solution = SCvxStarSolution(prob, size(u_ref,2))

    header = "\nIter |      J0      |    ΔJ_i     |    ΔL_i     |     χ_i     |     ρ_i     |    r_i    |     w     |  acpt. |"
    if verbosity > 0
        println(header)
    end

    for it in 1:maxiter
        # re-set non-convex expression according to reference
        if flag_reference
            if it > 1
                delete_noncvx_referencs!(prob, prob.model_nl_references)
            end
            g_dyn_ref, g_ref, h_ref = set_linearized_constraints!(prob, x_ref, u_ref, y_ref)
            set_trust_region_constraints!(algo, prob, x_ref, u_ref)   # if ref is updated, we need to update trust region constraints
        
        # only update trust-region constraints
        elseif flag_trust_region
            if it > 1
                delete_noncvx_referencs!(prob, [:constraint_trust_region_x_lb, :constraint_trust_region_x_ub])
            end
            set_trust_region_constraints!(algo, prob, x_ref, u_ref)   # if ref is not updated but trsut region size changed
        end

        # solve convex subproblem
        solve_convex_subproblem!(algo, prob)

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
        g_noncvx = prob.ng > 0 ? prob.g_noncvx(x_ref, u_ref, y_ref) : nothing
        h_noncvx = prob.nh > 0 ? max.(prob.h_noncvx(x_ref, u_ref, y_ref), 0) : nothing

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
            @printf(" %3.0f | % 1.5e | % 1.4e | % 1.4e | % 1.4e | % 1.4e | % 1.2e | % 1.2e |  %s   |\n",
                    it, J0, ΔJ, ΔL, χ, rho_i, algo.Δ, algo.w,
                    message_accept_step(rho_i >= algo.rhos[1]))
        end

        # update current solution
        solution.x[:,:] = _x
        solution.u[:,:] = _u
        solution.y = _y

        if store_iterates
            push!(solution.info[:J0], J0)
            push!(solution.info[:ΔJ], ΔJ)
            push!(solution.info[:χ], χ)
            push!(solution.info[:w], algo.w)
            push!(solution.info[:Δ], algo.Δ)
            push!(solution.info[:accept], rho_i >= algo.rhos[1])
            solution.n_iter += 1
        end

        if rho_i >= algo.rhos[1]
            if ((abs(ΔJ) <= tol_opt) && (χ <= tol_feas)) || ((J0 <= tol_J0) && (χ <= tol_feas))
                solution.status = :Optimal
                break
            end
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
        if solution.n_iter > 0
            @printf("   Objective                : %1.4e\n", solution.info[:J0][end])
            @printf("   Objective improvement ΔJ : %1.4e (tol: %1.4e)\n", solution.info[:ΔJ][end], tol_opt)
            @printf("   Max constraint violation : %1.4e (tol: %1.4e)\n", solution.info[:χ][end], tol_feas)
        end
        println()
    end
    return solution
end
