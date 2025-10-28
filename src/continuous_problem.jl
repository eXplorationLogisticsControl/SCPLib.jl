"""Continuous SCP problem"""


mutable struct ContinuousProblem <: OptimalControlProblem
    nx::Int
    nu::Int
    ny::Int
    N::Int
    Nu::Int

    ng::Int
    nh::Int

    eom!::Function
    eom_aug!::Function
    params

    times::Union{Vector,LinRange}

    objective::Function                     # (x,u,y) -> J0
    g_noncvx::Union{Function,Nothing}       # (x,u,y) -> g
    ∇g_noncvx::Union{Function,Nothing}      # (x,u,y) -> ∇g
    h_noncvx::Union{Function,Nothing}       # (x,u,y) -> h
    ∇h_noncvx::Union{Function,Nothing}      # (x,u,y) -> ∇h

    model::Model
    model_nl_references::Vector{Symbol}

    lincache::LinearizedCache

    ode_ensemble_method
    ode_method
    ode_reltol
    ode_abstol

    fun_get_trajectory::Union{Function,Nothing}
    set_dynamics_cache!::Union{Function,Nothing}

    u_bias::Matrix
end


function Base.show(io::IO, prob::ContinuousProblem)
    @printf("Continuous optimal control problem\n")
    @printf("  nx             : %d\n", prob.nx)
    @printf("  nu             : %d\n", prob.nu)
    @printf("  N              : %d\n", prob.N)
    @printf("  ng             : %d\n", prob.ng)
    @printf("  nh             : %d\n", prob.nh)
    @printf("  ODE ensemble   : %s\n", prob.ode_ensemble_method)
    @printf("  ODE method     : %s\n", prob.ode_method)
    @printf("  ODE reltol     : %1.4e\n", prob.ode_reltol)
    @printf("  ODE abstol     : %1.4e\n", prob.ode_abstol)
end


"""
Initialize augmented state for continuous dynamics
"""
function init_continuous_dynamics_xaug(x0::Vector, nx::Int, nu::Int)
    return [x0;
            reshape(I(nx), nx^2);
            zeros(nx * nu)];
end


function get_trajectory(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,i] + prob.u_bias[:,i]
        remake(ode_problem, u0=x_ref[:,i], tspan=(prob.times[i], prob.times[i+1]), p=_params)
    end
    base_ode_problem = ODEProblem(
        prob.eom!,
        x_ref[:,1],
        [0.0, 1.0],   # place holder
        prob.params,
    )
    ensemble_prob = EnsembleProblem(base_ode_problem, prob_func = prob_func)
    sols = solve(
        ensemble_prob,
        prob.ode_method,
        prob.ode_ensemble_method;
        trajectories = prob.N - 1,
        reltol = prob.ode_reltol,
        abstol = prob.ode_abstol,
    )
    for (k,sol) in enumerate(sols)
        g_dynamics[:,k] = x_ref[:,k+1] - sol.u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


"""
Propagate augmented dynamics with continuous control
This function also constructs state-transition matrices and 
evaluates dynamics residuals.
"""
function get_trajectory_augmented(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _x0_aug = init_continuous_dynamics_xaug(x_ref[:,i], prob.nx, prob.nu)
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,i] + prob.u_bias[:,i]
        remake(ode_problem, u0=_x0_aug, tspan=(prob.times[i], prob.times[i+1]), p=_params)
    end
    base_ode_problem = ODEProblem(
        prob.eom_aug!,
        init_continuous_dynamics_xaug(x_ref[:,1], prob.nx, prob.nu),
        [0.0, 1.0],   # place holder
        prob.params,
    )
    ensemble_prob = EnsembleProblem(base_ode_problem, prob_func = prob_func)
    sols = solve(
        ensemble_prob,
        prob.ode_method,
        prob.ode_ensemble_method;
        trajectories = prob.N - 1,
        reltol = prob.ode_reltol,
        abstol = prob.ode_abstol,
    )
    for (k,sol) in enumerate(sols)
        g_dynamics[:,k] = x_ref[:,k+1] - sol.u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


"""
Construct a continuous optimal control problem
"""
function ContinuousProblem(
    optimizer,
    eom!::Function,
    params,
    objective::Function,
    times,
    x_ref,
    u_ref,
    y_ref = nothing;
    eom_aug! = nothing,
    ng::Int = 0,
    g_noncvx::Union{Function,Nothing} = nothing,
    ∇g_noncvx::Union{Function,Nothing} = nothing,
    nh::Int = 0,
    h_noncvx::Union{Function,Nothing} = nothing,
    ∇h_noncvx::Union{Function,Nothing} = nothing,
    ode_ensemble_method = EnsembleSerial(),
    ode_method = Tsit5(),
    ode_reltol = 1e-12,
    ode_abstol = 1e-12,
    fun_get_trajectory::Union{Function,Nothing} = nothing,
    set_dynamics_cache!::Union{Function,Nothing} = nothing,
    u_bias::Union{Matrix,Nothing} = nothing,
)
    # get problem size from initial guess
    N = length(times)
    @assert size(x_ref,2) == size(u_ref,2) + 1 == N
    nx, _ = size(x_ref)
    nu, _ = size(u_ref)
    ny = isnothing(y_ref) ? 0 : length(y_ref)

    # construct augmented EOM using automatic differentiation
    if isnothing(eom_aug!)
        @warn "AD-based eom may be erroneous for now!"
        eom_aug! = get_continuous_augmented_eom(eom!, params, nx)
    end

    # initialize linearization cache
    lincache = LinearizedCache(nx, nu, N, N-1, ng, nh)

    # check if ∇g_noncvx is provided
    if !isnothing(g_noncvx) && isnothing(∇g_noncvx)
        ∇g_noncvx = function (x,u,y)
            return ForwardDiff.jacobian(z -> g_noncvx(unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u, y))
        end
    end

    # check if ∇h_noncvx is provided
    if !isnothing(h_noncvx) && isnothing(∇h_noncvx)
        ∇h_noncvx = function (x,u,y)
            return ForwardDiff.jacobian(z -> h_noncvx(unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u, y))
        end
    end

    # non-convex JuMP references
    model_nl_references = [:constraint_dynamics,
                           :constraint_trust_region_x_lb,
                           :constraint_trust_region_x_ub]


    if isnothing(u_bias)
        u_bias = zeros(nu,N)
    else
        @assert size(u_bias) == (nu,N)
    end

    # construct problem
    prob = ContinuousProblem(
        nx,
        nu,
        ny,
        N,
        N-1,
        ng,
        nh,
        eom!,
        eom_aug!,
        params,
        times,
        objective,
        g_noncvx,
        ∇g_noncvx,
        h_noncvx,
        ∇h_noncvx,
        Model(optimizer),
        model_nl_references,
        lincache,
        ode_ensemble_method,
        ode_method,
        ode_reltol,
        ode_abstol,
        fun_get_trajectory,
        set_dynamics_cache!,
        u_bias,
    )

    # poopulate JuMP with variables
    @variable(prob.model, x[i=1:nx, k=1:N])
    @variable(prob.model, u[i=1:nu, k=1:N-1])
    @variable(prob.model, ξ_dyn[i=1:nx, k=1:N-1])

    if ng > 0
        @variable(prob.model, ξ[i=1:ng])
        push!(prob.model_nl_references, :constraint_g_noncvx)
    end

    if nh > 0
        @variable(prob.model, ζ[i=1:nh] >= 0.0)
        push!(prob.model_nl_references, :constraint_h_noncvx)
    end
    return prob
end


function stack_flatten_variables(prob::ContinuousProblem, x, u, y)
    Δz = [reshape(x, prob.nx * prob.N);
          reshape(u, prob.nu * (prob.N-1))];
    if prob.ny > 0
        Δz = [Δz; y]
    end
    return Δz
end


function unpack_flattened_variables(prob::ContinuousProblem, z)
    x = reshape(z[1:prob.nx * prob.N], prob.nx, prob.N)
    u = reshape(z[prob.nx * prob.N + 1:prob.nx * prob.N + prob.nu * (prob.N-1)], prob.nu, prob.N-1)
    if prob.ny > 0
        y = z[prob.nx * prob.N + prob.nu * (prob.N-1) + 1:end]
    else
        y = nothing
    end
    return x, u, y
end