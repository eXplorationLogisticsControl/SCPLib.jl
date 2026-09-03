"""Utility functions for STM"""


function get_stm_adode(
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

    # compute STM using ForwardDiff
    result = DiffResults.JacobianResult(x0)
    ForwardDiff.jacobian!(result, _solve_ode, x0)
    return result
end