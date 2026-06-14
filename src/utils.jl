"""Utility functions"""


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
    for ref in unique(references)
        delete_model_reference!(prob, ref)
    end
end


"""Remove a registered JuMP model reference if it currently exists."""
function delete_existing_references!(prob::OptimalControlProblem, references::Vector{Symbol})
    model_objects = object_dictionary(prob.model)
    for ref in unique(references)
        if haskey(model_objects, ref)
            delete_model_reference!(prob, ref)
        end
    end
    return
end


function delete_model_reference!(prob::OptimalControlProblem, ref::Symbol)
    delete(prob.model, prob.model[ref])
    unregister(prob.model, ref)
    return
end


"""Append model references without creating duplicate cleanup entries."""
function append_unique_references!(references::Vector{Symbol}, new_references::Vector{Symbol})
    for ref in new_references
        if ref ∉ references
            push!(references, ref)
        end
    end
    return
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