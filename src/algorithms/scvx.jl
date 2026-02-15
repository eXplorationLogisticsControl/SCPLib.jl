"""SCvx algorithm"""


"""
SCvx algorithm

# Arguments
"""
mutable struct SCvx <: TrustRegionAlgorithm
    # storage
    tr::TrustRegions

    # algorithm parameters
    w::Float64
    rhos::Tuple{Real,Real,Real}
    alpha::Float64
    Δ_bounds::Tuple{Float64,Float64}
    beta::Float64

    function SCvx(
        nx::Int,
        N::Int;
        w::Float64 = 1e0,
        rhos::Tuple{Real,Real,Real} = (0.0, 0.25, 0.7),
        alpha::Float64 = 2.0,
        Δ_bounds::Tuple{Float64,Float64} = (1e-6, 1e4),
        beta::Float64 = 2.0,
    )
        tr = TrustRegions(nx, N, Δ0)
        new(tr, w, rhos, alpha, Δ_bounds, beta)
    end
end


function Base.show(io::IO, algo::SCvx)
    println(io, "SCvx algorithm")
    @printf("   Trust-region size Δ                                : %1.2e\n", algo.tr.Δ[1,1])
    @printf("   Penalty weight w                                   : %1.2e\n", algo.w)
    @printf("   Penalty weight update factor β                     : %1.2e\n", algo.beta)
    @printf("   Trust-region size bounds (Δ_min, Δ_max)            : %1.2e, %1.2e\n", algo.Δ_bounds[1], algo.Δ_bounds[2])
end


"""
SCvx exact penalty function (l1-penalty)
"""
function penalty(algo::SCvx, prob::OptimalControlProblem, ξ_dyn, ξ, ζ)
    P = sqrt(algo.w) * norm(ξ_dyn,1)
    if prob.ng > 0
        P += sqrt(algo.w) * norm(ξ,1)
    end
    if prob.nh > 0
        P += sqrt(algo.w) * norm(ζ,1)
    end
    return P
end


function penalty(algo::SCvx, prob::OptimalControlProblem, ξ_dyn::Matrix{VariableRef}, ξ, ζ, slacks_L1)
    P = sqrt(algo.w) * sum(slacks_L1[:slack_gdyn])
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
    return P
end


"""Solve convex subproblem for SCvx algorithm"""
function solve_convex_subproblem!(algo::SCvx, prob::OptimalControlProblem)
    # set objective with penalty
    _ξ = prob.ng > 0 ? prob.model[:ξ] : nothing
    _ζ = prob.nh > 0 ? prob.model[:ζ] : nothing

    # append slack variable for l1 penalty term
    slacks_L1 = Dict(
        :slack_gdyn => @variable(prob.model),
        :slack_gnoncvx => prob.ng > 0 ? @variable(prob.model) : nothing,
        :slack_hnoncvx => prob.nh > 0 ? @variable(prob.model) : nothing,
    )

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
        u = zeros(prob.nu, Nu)
        
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


"""Solve non-convex OCP with SCvx algorithm"""
function solve!(
    algo::SCvx,
    prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint};
    maxiter::Int = 100,
    tol_feas::Float64 = 1e-6,
    tol_opt::Float64 = 1e-6,
    verbosity::Int = 1,
    store_iterates::Bool = false,
    callback::Union{Nothing,Function} = nothing,
    warmstart_primal::Bool = false,
    warmstart_dual::Bool = false,
)
    @assert prob.ng == length(algo.λ) "Number of non-convex equality constraints mismatch between problem and algorithm"
    @assert prob.nh == length(algo.μ) "Number of non-convex inequality constraints mismatch between problem and algorithm"
    tcpu_start = time()

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
        @printf("   Initial penalty weight w       : % 1.2e\n", algo.w)
        println(header)
    end

    # initialize warmstart dictionaries
    variable_primal = Dict()
    constraint_solution = Dict()

    for it in 1:maxiter

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
