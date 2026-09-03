"""Utility functions for STM"""


"""
Compute STM using ForwardDiff through ODE integration

Extract final state via `DiffResults.value(result)` and
Jacobian via `DiffResults.jacobian(result)` where `result` is 
the output of `get_stm_x_adode`.
"""
function get_stm_x_adode(
    eom!,
    x0,
    tspan,
    params,
    method=Vern7();
    kwargs...
)
    # wrapper function to return final state
    _solve_ode = function (_x0)
        sol = solve(ODEProblem(eom!, _x0, tspan, params), method; kwargs...)
        return sol.u[end]
    end

    # compute STM with respect to state using ForwardDiff
    result = DiffResults.JacobianResult(x0)
    ForwardDiff.jacobian!(result, _solve_ode, x0)
    return result
end


"""
Compute STM with respect to control using ForwardDiff through ODE integration.

`params` is the ODE parameter `p = (; params, u)`. The Jacobian is `∂x(tf)/∂u`.

Extract final state via `DiffResults.value(result)` and
Jacobian via `DiffResults.jacobian(result)` where `result` is 
the output of `get_stm_u_adode`.
"""
function get_stm_u_adode(
    eom!,
    x0,
    u0,
    tspan,
    params,
    method=Vern7();
    kwargs...
)
    # wrapper function
    _solve_ode = function (_u0)
        T = eltype(_u0)
        sol = solve(
            ODEProblem(eom!, T.(x0), tspan, dynamics_input(params.params, _u0)),
            method;
            kwargs...,
        )
        return sol.u[end]
    end

    # compute STM with respect to control using ForwardDiff
    result = DiffResults.JacobianResult(x0, u0)
    ForwardDiff.jacobian!(result, _solve_ode, u0)
    return result
end


"""
Compute STM with respect to state and control using ForwardDiff through ODE integration.

`params` is the ODE parameter `p = (; params, u)`. The Jacobian is `∂x(tf)/∂[x0; u]`.

Extract final state via `DiffResults.value(result)` and
Jacobian via `DiffResults.jacobian(result)` where `result` is
the output of `get_stm_adode`.
"""
function get_stm_adode(
    eom!,
    x0,
    u0,
    tspan,
    params,
    method=Vern7();
    kwargs...
)
    nx = length(x0)
    physics_params = params.params
    xu0 = [x0; u0]

    _solve_ode = function (_xu0)
        _x0 = _xu0[1:nx]
        _u0 = _xu0[nx+1:end]
        sol = solve(
            ODEProblem(eom!, _x0, tspan, dynamics_input(physics_params, _u0)),
            method;
            kwargs...,
        )
        return sol.u[end]
    end

    result = DiffResults.JacobianResult(x0, xu0)
    ForwardDiff.jacobian!(result, _solve_ode, xu0)
    return result
end