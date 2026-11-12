"""Utility functions"""


"""
Solver verbosity for `OrdinaryDiffEq`.

v6 accepted `verbose = false`. v7 requires a `DEVerbosity` object.
"""
function ode_verbose()
    if isdefined(OrdinaryDiffEq, :DEVerbosity)
        return OrdinaryDiffEq.DEVerbosity(OrdinaryDiffEq.SciMLLogging.None())
    else
        return false
    end
end


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


# Names that `solve!` registers in `prob.model`, in addition to the user's entries in
# `prob.model_nl_references`. `reset_solver_model!` removes these at the start of each solve.
const SOLVER_REFERENCES = [
    :constraint_dynamics,
    :constraint_g_noncvx,
    :constraint_h_noncvx,
    :constraint_trust_region_x_lb,
    :constraint_trust_region_x_ub,
    :constraint_trust_region_u_lb,
    :constraint_trust_region_u_ub,
    :constraint_slack_gdyn,
    :constraint_slack_gncvx,
    :constraint_slack_hncvx,
]


# Key in `prob.model.ext` that holds the unnamed variables and constraints added by `solve!`
const SOLVER_OBJECTS_KEY = :SCPLib_solver_objects


"""
    track_solver_object!(prob, obj)

Record `obj` so that the next `solve!` on `prob` deletes it. Use this for unnamed variables and
constraints that `solve!` adds once per solve, such as the slack variables of the penalty terms.
`obj` can be a variable or constraint reference, or an array or `Dict` of them; if it is mutable,
the references it holds when the next solve starts are deleted. Returns `obj`.
"""
function track_solver_object!(prob::OptimalControlProblem, obj)
    push!(get!(prob.model.ext, SOLVER_OBJECTS_KEY, Any[]), obj)
    return obj
end


"""Collect the variable and constraint references in `obj` (a reference, or an array or `Dict` of them)"""
function collect_references!(refs::Vector{Any}, obj)
    if obj isa ConstraintRef || obj isa VariableRef
        push!(refs, obj)
    elseif obj isa AbstractDict
        foreach(o -> collect_references!(refs, o), values(obj))
    elseif obj isa AbstractArray
        foreach(o -> collect_references!(refs, o), obj)
    end
    return refs
end


"""
Delete the references in `objs` from `model`, skipping those already deleted. Constraints are
deleted before variables, as a solver may not allow deleting a variable that is still in a
vector constraint (e.g. `[t; ξ] in MOI.NormOneCone`).
"""
function delete_if_valid!(model::Model, objs...)
    refs = Any[]
    foreach(o -> collect_references!(refs, o), objs)
    for ref in Iterators.flatten((filter(r -> r isa ConstraintRef, refs), filter(r -> r isa VariableRef, refs)))
        is_valid(model, ref) && delete(model, ref)
    end
    return
end


"""
    reset_solver_model!(prob)

Remove from `prob.model` everything that an earlier `solve!` added: the linearized and
trust-region constraints, the penalty slack constraints, and the unnamed slack variables
recorded with `track_solver_object!`. After this, the model holds only what the user defined,
so `solve!` can be called again on the same problem, also with a different algorithm, or after
a solve that threw an error.
"""
function reset_solver_model!(prob::OptimalControlProblem)
    model = prob.model
    for ref in unique(vcat(SOLVER_REFERENCES, prob.model_nl_references))
        if haskey(object_dictionary(model), ref)
            delete_if_valid!(model, model[ref])
            unregister(model, ref)
        end
    end
    delete_if_valid!(model, pop!(model.ext, SOLVER_OBJECTS_KEY, Any[]))
    return
end


"""
    linearization_references(prob, algo)

Names of the constraints that `algo` deletes and re-creates when the reference changes: the
problem's `prob.model_nl_references`, plus the control trust-region constraints when `algo`
uses them. Returns a new vector, so `prob.model_nl_references` is never modified.
"""
function linearization_references(prob::OptimalControlProblem, algo::SCPAlgorithm)
    refs = copy(prob.model_nl_references)
    if hasproperty(algo, :use_trustregion_control) && algo.use_trustregion_control
        for ref in (:constraint_trust_region_u_lb, :constraint_trust_region_u_ub)
            ref in refs || push!(refs, ref)
        end
    end
    return refs
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


"""
Set optimal start values to JuMP model

Entries whose variable or constraint has since been deleted are skipped: re-linearizing
deletes the linearized and trust-region constraints that were stored at the accepted step.
"""
function set_optimal_start_values(variable_primal::Dict, constraint_solution::Dict)
    for (x, primal_start) in variable_primal
        is_valid(owner_model(x), x) || continue
        set_start_value(x, primal_start)
    end
    for (ci, (primal_start, dual_start)) in constraint_solution
        is_valid(owner_model(ci), ci) || continue
        set_start_value(ci, primal_start)
        set_dual_start_value(ci, dual_start)
    end
    return
end

"""Three-character step-acceptance message for the iteration log"""
function message_step(kind::Symbol, accept::Bool)
    if kind == :restoration
        return "rst"
    elseif kind == :stationary
        return "nul"
    elseif kind == :unresolved
        return "exp"
    else
        return message_accept_step(accept)
    end
end


"""Maximum constraint violation χ from dynamics defects and (clipped) non-convex constraint values"""
function constraint_violation(prob::OptimalControlProblem, g_dynamics, g_noncvx, h_noncvx)
    χ = norm(g_dynamics, Inf)
    if prob.ng > 0
        χ = max(χ, norm(g_noncvx, Inf))
    end
    if prob.nh > 0
        χ = max(χ, norm(h_noncvx, Inf))
    end
    return χ
end
