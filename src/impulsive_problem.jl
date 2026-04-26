"""Impulsive SCP problem"""


mutable struct ImpulsiveProblem <: OptimalControlProblem
    nx::Int
    nu::Int
    N::Int
    Nu::Int

    ng::Int
    nh::Int

    eom!::Function
    eom_aug!::Function
    dfdu::Function
    params

    times::Union{Vector,LinRange}

    objective::Function                     # (x,u) -> J0
    g_noncvx::Union{Function,Nothing}       # (cache,x,u) -> g
    ∇g_noncvx::Union{Function,Nothing}      # (cache,x,u) -> ∇g
    h_noncvx::Union{Function,Nothing}       # (cache,x,u) -> h
    ∇h_noncvx::Union{Function,Nothing}      # (cache,x,u) -> ∇h

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


function Base.show(io::IO, prob::ImpulsiveProblem)
    @printf("Impulsive optimal control problem\n")
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
function init_impulsive_dynamics_xaug(t0::Real, x0::Vector, u0::Vector, nx::Int, dfdu::Function)
    return [x0 + dfdu(x0, u0, t0)*u0;
            reshape(I(nx), nx^2)];
end


function get_trajectory(prob::ImpulsiveProblem, x_ref, u_ref)
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _params = deepcopy(prob.params)
        remake(ode_problem, u0=x_ref[:,i] + prob.dfdu(x_ref[:,i], u_ref[:,i], prob.times[i]) * (u_ref[:,i] + prob.u_bias[:,i]),
            tspan=(prob.times[i], prob.times[i+1]), p=_params)
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
function get_trajectory_augmented(prob::ImpulsiveProblem, x_ref, u_ref)
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _x0_aug = init_impulsive_dynamics_xaug(prob.times[i], x_ref[:,i], u_ref[:,i] + prob.u_bias[:,i], prob.nx, prob.dfdu)
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,i] + prob.u_bias[:,i]
        remake(ode_problem, u0=_x0_aug, tspan=(prob.times[i], prob.times[i+1]), p=_params)
    end
    base_ode_problem = ODEProblem(
        prob.eom_aug!,
        [x_ref[:,1]; reshape(I(prob.nx), prob.nx^2)],
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
Construct an impulsive optimal control problem

If `set_dynamics_cache!` is provided, it will be used to set STM's to `prob.lincache`.
The signature of `set_dynamics_cache!` should be:
```julia
set_dynamics_cache!(prob::OptimalControlProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
```
and the function should return the dynamics residuals. 
See `set_dynamics_cache!` for more details.

# Arguments:
- `optimizer`: optimizer for the JuMP model
- `eom!::Function`: impulsive dynamics function
- `params`: parameters for the problem
- `objective::Function`: objective function
- `times`: time points
- `x_ref`: reference state
- `u_ref`: reference control
- `eom_aug!::Function`: augmented dynamics function
- `dfdu::Function`: derivative of the impulsive dynamics function with respect to the control
- `ng::Int`: number of non-convex equality constraints
- `g_noncvx::Union{Function,Nothing}`: non-convex equality constraints with signature `(lincache, x, u) -> g`
- `∇g_noncvx::Union{Function,Nothing}`: gradient of non-convex equality constraints with signature `(lincache, x, u) -> ∇g`
- `nh::Int`: number of non-convex inequality constraints
- `h_noncvx::Union{Function,Nothing}`: non-convex inequality constraints with signature `(lincache, x, u) -> h`
- `∇h_noncvx::Union{Function,Nothing}`: gradient of non-convex inequality constraints with signature `(lincache, x, u) -> ∇h`
- `ode_ensemble_method`: ensemble method for the ODE solver
- `ode_method`: method for the ODE solver
- `ode_reltol`: relative tolerance for the ODE solver
- `ode_abstol`: absolute tolerance for the ODE solver
- `fun_get_trajectory::Union{Function,Nothing}`: user-defined function to get the trajectory
- `set_dynamics_cache!::Union{Function,Nothing}`: user-defined function to set the dynamics cache
- `u_bias::Union{Matrix,Nothing}`: bias on the control
"""
function ImpulsiveProblem(
    optimizer,
    eom!::Function,
    params,
    objective::Function,
    times,
    x_ref,
    u_ref;
    eom_aug! = nothing,
    dfdu::Function = (x,u,t) -> [zeros(3,4); I(3) zeros(3,1)],
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
    # get problem size from initial guess
    N = length(times)
    Nu = size(u_ref,2)
    @assert Nu in [N-1, N]
    nx, _ = size(x_ref)
    nu, _ = size(u_ref)

    # check on size of dfdu function
    _dfdu_test = dfdu(x_ref[:,1], u_ref[:,1], times[1])
    @assert size(_dfdu_test) == (nx, nu) "Size of output from dfdu is not (nx,nu)"

    # construct augmented EOM using automatic differentiation
    if isnothing(eom_aug!)
        eom_aug! = get_impulsive_augmented_eom(eom!, params, nx)
    end

    # initialize linearization cache
    lincache = MultipleShootingCache(nx, nu, N, Nu, ng, nh)

    # check if ∇g_noncvx is provided
    if !isnothing(g_noncvx) && isnothing(∇g_noncvx)
        ∇g_noncvx = function (x,u)
            return ForwardDiff.jacobian(z -> g_noncvx(lincache, unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u))
        end
    end

    # check if ∇h_noncvx is provided
    if !isnothing(h_noncvx) && isnothing(∇h_noncvx)
        ∇h_noncvx = function (x,u)
            return ForwardDiff.jacobian(z -> h_noncvx(lincache, unpack_flattened_variables(prob, z)...),
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

    # construct problem struct
    prob = ImpulsiveProblem(
        nx,
        nu,
        N,
        Nu,
        ng,
        nh,
        eom!,
        eom_aug!,
        dfdu,
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
    @variable(prob.model, x[i=1:nx, k=1:N])
    @variable(prob.model, u[i=1:nu, k=1:Nu])
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


function stack_flatten_variables(prob::ImpulsiveProblem, x, u)
    Δz = [reshape(x, prob.nx * prob.N);
          reshape(u, prob.nu * prob.Nu)];
    return Δz
end


function unpack_flattened_variables(prob::ImpulsiveProblem, z)
    x = reshape(z[1:prob.nx * prob.N], prob.nx, prob.N)
    u = reshape(z[prob.nx * prob.N + 1:prob.nx * prob.N + prob.nu * prob.Nu], prob.nu, prob.Nu)
    return x, u
end


