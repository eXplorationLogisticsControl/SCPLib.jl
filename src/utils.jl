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
    for ref in references
        delete(prob.model, prob.model[ref])
        unregister(prob.model, ref)
    end
end
