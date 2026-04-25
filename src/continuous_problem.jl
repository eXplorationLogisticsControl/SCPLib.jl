"""Continuous SCP problem"""


mutable struct ContinuousProblem <: OptimalControlProblem
    nx::Int
    nu::Int
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

    lincache::AbstractLinearizationCache

    ode_ensemble_method
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64

    fun_get_trajectory::Union{Function,Nothing}
    set_dynamics_cache!::Union{Function,Nothing}

    u_bias::Matrix
    shooting_method::Symbol
end


function Base.show(io::IO, prob::ContinuousProblem)
    @printf("Continuous optimal control problem\n")
    @printf("  Shooting method : %s\n", prob.shooting_method)
    @printf("  nx              : %d\n", prob.nx)
    @printf("  nu              : %d\n", prob.nu)
    @printf("  N               : %d\n", prob.N)
    @printf("  ng              : %d\n", prob.ng)
    @printf("  nh              : %d\n", prob.nh)
    @printf("  ODE ensemble    : %s\n", prob.ode_ensemble_method)
    @printf("  ODE method      : %s\n", prob.ode_method)
    @printf("  ODE reltol      : %1.4e\n", prob.ode_reltol)
    @printf("  ODE abstol      : %1.4e\n", prob.ode_abstol)
end


"""
Initialize augmented state for continuous dynamics
"""
function init_continuous_dynamics_xaug(x0::Vector, nx::Int, nu::Int)
    return [x0;
            reshape(I(nx), nx^2);
            zeros(nx * nu)];
end


function get_trajectory(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
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
function get_trajectory_augmented(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
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


function get_trajectory_forwardbackward(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
    # define base ODE & storage
    base_ode_problem = ODEProblem(
        prob.eom!,
        x_ref[:,1],
        [0.0, 1.0],   # place holder
        prob.params,
    )
    sols = Vector{ODESolution}(undef, prob.N-1)
    
    # forward shooting
    Nu_fwd = div(prob.N, 2)
    xk_fwd = deepcopy(x_ref[:,1])
    for k in 1:Nu_fwd
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,k] + prob.u_bias[:,k]
        ode_problem = remake(base_ode_problem, u0=xk_fwd, tspan=(prob.times[k], prob.times[k+1]), p=_params)
        sols[k] = solve(ode_problem, prob.ode_method; reltol = prob.ode_reltol, abstol = prob.ode_abstol)
        xk_fwd = sols[k].u[end][1:prob.nx]
    end
    
    # backward shooting
    xk_bwd = deepcopy(x_ref[:,end])
    for k in prob.N-1:-1:Nu_fwd+1
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,k] + prob.u_bias[:,k]
        ode_problem = remake(base_ode_problem, u0=xk_bwd, tspan=(prob.times[k+1], prob.times[k]), p=_params)
        sols[k] = solve(ode_problem, prob.ode_method; reltol = prob.ode_reltol, abstol = prob.ode_abstol)
        xk_bwd = sols[k].u[end][1:prob.nx]
    end
    g_dynamics = zeros(prob.nx,1)
    g_dynamics[:,1] = xk_bwd - xk_fwd
    return sols, g_dynamics
end


"""
Propagate augmented dynamics with continuous control
This function also constructs state-transition matrices and evaluates dynamics residual at the match point.
"""
function get_trajectory_augmented_forwardbackward(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
    # define base ODE & storage
    base_ode_problem = ODEProblem(
        prob.eom_aug!,
        init_continuous_dynamics_xaug(x_ref[:,1], prob.nx, prob.nu),
        [0.0, 1.0],   # place holder
        prob.params,
    )
    sols = Vector{ODESolution}(undef, prob.N-1)
    
    # forward shooting
    Nu_fwd = div(prob.N, 2)
    xk_fwd = deepcopy(x_ref[:,1])
    for k in 1:Nu_fwd
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,k] + prob.u_bias[:,k]
        ode_problem = remake(base_ode_problem, u0=init_continuous_dynamics_xaug(xk_fwd, prob.nx, prob.nu), tspan=(prob.times[k], prob.times[k+1]), p=_params)
        sols[k] = solve(ode_problem, prob.ode_method; reltol = prob.ode_reltol, abstol = prob.ode_abstol)
        xk_fwd = sols[k].u[end][1:prob.nx]
    end
    
    # backward shooting
    xk_bwd = deepcopy(x_ref[:,end])
    for k in prob.N-1:-1:Nu_fwd+1
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,k] + prob.u_bias[:,k]
        ode_problem = remake(base_ode_problem, u0=init_continuous_dynamics_xaug(xk_bwd, prob.nx, prob.nu), tspan=(prob.times[k+1], prob.times[k]), p=_params)
        sols[k] = solve(ode_problem, prob.ode_method; reltol = prob.ode_reltol, abstol = prob.ode_abstol)
        xk_bwd = sols[k].u[end][1:prob.nx]
    end
    g_dynamics = zeros(prob.nx,1)
    g_dynamics[:,1] = xk_bwd - xk_fwd
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
    u_ref;
    eom_aug! = nothing,
    ng::Int = 0,
    g_noncvx::Union{Function,Nothing} = nothing,
    ∇g_noncvx::Union{Function,Nothing} = nothing,
    nh::Int = 0,
    h_noncvx::Union{Function,Nothing} = nothing,
    ∇h_noncvx::Union{Function,Nothing} = nothing,
    ode_ensemble_method = EnsembleSerial(),
    ode_method = Tsit5(),
    ode_reltol::Float64 = 1e-12,
    ode_abstol::Float64 = 1e-12,
    fun_get_trajectory::Union{Function,Nothing} = nothing,
    set_dynamics_cache!::Union{Function,Nothing} = nothing,
    u_bias::Union{Matrix,Nothing} = nothing,
    shooting_method::Symbol = :multiple,
)
    @assert shooting_method in [:multiple, :forwardbackward]

    # get problem size from initial guess
    N = length(times)
    if shooting_method == :multiple
        @assert size(x_ref,2) == N
    else
        @assert size(x_ref,2) == 2
    end
    @assert size(u_ref,2) == N - 1
    nx, _ = size(x_ref)
    nu, _ = size(u_ref)

    # construct augmented EOM using automatic differentiation
    if isnothing(eom_aug!)
        eom_aug! = get_continuous_augmented_eom(eom!, params, nx)
    end

    # initialize linearization cache
    if shooting_method == :multiple
        lincache = MultipleShootingCache(nx, nu, N, N-1, ng, nh)
    elseif shooting_method == :forwardbackward
        lincache = ForwardBackwardCache(nx, nu, N, N-1, ng, nh)
    else
        @error "Invalid shooting method: $shooting_method"
    end

    # check if ∇g_noncvx is provided
    if !isnothing(g_noncvx) && isnothing(∇g_noncvx)
        ∇g_noncvx = function (x,u)
            return ForwardDiff.jacobian(z -> g_noncvx(unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u))
        end
    end

    # check if ∇h_noncvx is provided
    if !isnothing(h_noncvx) && isnothing(∇h_noncvx)
        ∇h_noncvx = function (x,u)
            return ForwardDiff.jacobian(z -> h_noncvx(unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u))
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
        shooting_method,
    )

    # poopulate JuMP with variables
    @variable(prob.model, u[i=1:nu, k=1:N-1])
    if shooting_method == :multiple
        @variable(prob.model, x[i=1:nx, k=1:N])
        @variable(prob.model, ξ_dyn[i=1:nx, k=1:N-1])
    elseif shooting_method == :forwardbackward
        @variable(prob.model, x[i=1:nx, k=1:2])
        @variable(prob.model, ξ_dyn[i=1:nx, k=1:1])
    end

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


function stack_flatten_variables(prob::ContinuousProblem, x, u)
    Δz = [reshape(x, prob.nx * prob.N);
          reshape(u, prob.nu * (prob.N-1))];
    return Δz
end


function unpack_flattened_variables(prob::ContinuousProblem, z)
    x = reshape(z[1:prob.nx * prob.N], prob.nx, prob.N)
    u = reshape(z[prob.nx * prob.N + 1:prob.nx * prob.N + prob.nu * (prob.N-1)], prob.nu, prob.N-1)
    return x, u
end