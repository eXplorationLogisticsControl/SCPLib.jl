"""Continuous SCP problem"""


mutable struct ContinuousProblem <: OptimalControlProblem
    nx::Int
    nu::Int
    ny::Int
    N::Int

    ng::Int
    nh::Int

    eom!::Function
    eom_aug!::Function
    params

    times::Union{Vector,LinRange}

    objective::Function                     # (x,u,y) -> J
    g_noncvx::Union{Function,Nothing}
    h_noncvx::Union{Function,Nothing}

    model::Model
    model_nl_references::Vector{Symbol}

    lincache::LinearizedCache

    ode_ensemble_method
    ode_method
    ode_reltol
    ode_abstol
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
Remove non-convex constraints from model within `ContinuousProblem`'s JuMP model
"""
function delete_noncvx_referencs!(prob::ContinuousProblem, references::Vector{Symbol})
    for ref in references
        delete(prob.model, prob.model[ref])
        unregister(prob.model, ref)
    end
end


"""
Initialize augmented state for continuous dynamics
"""
function initialize_augmented_state(x0::Vector, nx::Int, nu::Int)
    return [x0;
            reshape(I(nx), nx^2);
            zeros(nx * nu)];
end


function get_trajectory(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,i]
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
    for k in 1:prob.N-1
        g_dynamics[:,k] = x_ref[:,k+1] - sols[k].u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


function get_trajectory_augmented(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    g_dynamics = zeros(prob.nx, prob.N-1)
    # set dynamics constraints
    prob_func = function(ode_problem, i, repeat)
        _x0_aug = initialize_augmented_state(x_ref[:,i], prob.nx, prob.nu)
        _params = deepcopy(prob.params)
        _params.u[:] = u_ref[:,i]
        remake(ode_problem, u0=_x0_aug, tspan=(prob.times[i], prob.times[i+1]), p=_params)
    end
    base_ode_problem = ODEProblem(
        prob.eom_aug!,
        initialize_augmented_state(x_ref[:,1], prob.nx, prob.nu),
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
    for k in 1:prob.N-1
        g_dynamics[:,k] = x_ref[:,k+1] - sols[k].u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


function evaluate_g_noncvx(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    g_noncvx = nothing
    # FIXME
    error("evaluate_g_noncvx not implemented")
    return g_noncvx
end


function evaluate_h_noncvx(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    h_noncvx = nothing
    # FIXME
    error("evaluate_h_noncvx not implemented")
    return h_noncvx
end


"""
Set non-convex expressions
"""
function set_noncvx_expressions!(prob::ContinuousProblem, x_ref, u_ref, y_ref)
    # set dynamics constraints
    sols, g_dynamics = get_trajectory_augmented(prob, x_ref, u_ref, y_ref)
    set_dynamics_cache!(prob.lincache, x_ref, u_ref, sols)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )

    # set nonconvex equality constraints
    g_noncvx = nothing
    # FIXME

    # set nonconvex inequality constraints
    h_noncvx = nothing
    # FIXME
    return g_dynamics, g_noncvx, h_noncvx
end


"""
Construct a continuous control problem
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
    nh::Int = 0,
    h_noncvx::Union{Function,Nothing} = nothing,
    ode_ensemble_method = EnsembleSerial(),
    ode_method = Tsit5(),
    ode_reltol = 1e-12,
    ode_abstol = 1e-12,
)
    # get problem size from initial guess
    N = length(times)
    @assert size(x_ref,2) == size(u_ref,2) + 1 == N
    nx, _ = size(x_ref)
    nu, _ = size(u_ref)

    if !isnothing(y_ref)
        ny = length(y_ref)
    else
        ny = 0
    end

    # construct augmented EOM using automatic differentiation
    if isnothing(eom_aug!)
        eom_aug! = function (dx_aug, x_aug, params, t)
            # FIXME
            error("AD definition of eom_aug! not yet implemented")
        end
    end

    # initialize linearization cache
    lincache = LinearizedCache(nx, nu, N, ng, nh)

    # construct problem
    model_nl_references = [:constraint_dynamics, :constraint_trust_region_x_lb, :constraint_trust_region_x_ub]
    prob = ContinuousProblem(
        nx,
        nu,
        ny,
        N,
        ng,
        nh,
        eom!,
        eom_aug!,
        params,
        times,
        objective,
        g_noncvx,
        h_noncvx,
        Model(optimizer),
        model_nl_references,
        lincache,
        ode_ensemble_method,
        ode_method,
        ode_reltol,
        ode_abstol,
    )

    # poopulate JuMP with variables
    @variable(prob.model, x[i=1:nx, k=1:N])
    @variable(prob.model, u[i=1:nu, k=1:N-1])
    @variable(prob.model, ξ_dyn[i=1:nx, k=1:N-1])

    if ng > 0
        @variable(prob.model, ξ[i=1:ng])
        push!(prob.model_nl_references, :g_noncvx)
    end

    if nh > 0
        @variable(prob.model, ζ[i=1:nh])
        push!(prob.model_nl_references, :h_noncvx)
    end
    return prob
end
