"""Utility functions"""


"""
Trajectory index for `EnsembleProblem` `prob_func`.

OrdinaryDiffEq v6 / SciMLBase called `prob_func(prob, i, repeat)` with integer `i`.
OrdinaryDiffEq v7 calls `prob_func(prob, ctx)` with `ctx.sim_id`.
"""
ensemble_sim_id(i::Integer) = i
ensemble_sim_id(ctx) = ctx.sim_id


"""Per-trajectory solutions from an ensemble (`sols.u`) or a plain vector."""
ensemble_trajectories(sols::SciMLBase.AbstractEnsembleSolution) = sols.u
ensemble_trajectories(sols::AbstractVector) = sols


function vprintf(verbosity::Int, str::String)
    if verbosity > 0
        println(str)
    end
end


function message_accept_step(accept::Bool)
    if accept
        return "yes"
    else
        return "no "
    end
end


"""
Remove non-convex constraints from model within `OptimalControlProblem`'s JuMP model
"""
function delete_noncvx_referencs!(prob::OptimalControlProblem, references::Vector{Symbol})
    for ref in references
        cons = prob.model[ref]
        if cons isa ConstraintRef
            delete(prob.model, cons)
        else
            for c in cons
                delete(prob.model, c)
            end
        end
        unregister(prob.model, ref)
    end
end


"""Extract primal variables from JuMP model"""
function get_primal_variables(model::Model)
    return Dict(x => value(x) for x in all_variables(model))
end


"""Extract constraint solutions from JuMP model"""
function get_constraint_solutions(model::Model)
    constraint_solution = Dict()
    for (F, S) in list_of_constraint_types(model)
        try
            for ci in all_constraints(model, F, S)
                constraint_solution[ci] = (value(ci), dual(ci))
            end
        catch
            @info("Something went wrong getting $F-in-$S. Skipping")
        end
    end
    return constraint_solution
end


"""Set optimal start values to JuMP model"""
function set_optimal_start_values(variable_primal::Dict, constraint_solution::Dict)
    for (x, primal_start) in variable_primal
        set_start_value(x, primal_start)
    end
    for (ci, (primal_start, dual_start)) in constraint_solution
        set_start_value(ci, primal_start)
        set_dual_start_value(ci, dual_start)
    end
    return
end