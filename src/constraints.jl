"""Setting contraints to convex subproblem"""


"""
Set linearized dynamics constraints for continuous problem
"""
function set_dynamics_cache!(
    prob::ContinuousProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint}
)
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref)
    set_continuous_dynamics_cache!(prob.lincache, x_ref, u_ref, sols)
    return g_dynamics_ref
end


"""
Set linearized dynamics constraints for impulsive problem
"""
function set_dynamics_cache!(
    prob::ImpulsiveProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint},
)
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref)
    set_impulsive_dynamics_cache!(prob.lincache, x_ref, u_ref, sols, prob.dfdu)
    return g_dynamics_ref
end


"""
Set linearized constraints for non-convex constraints
"""
function set_linearized_constraints!(
    prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint},
)
    # set dynamics constraints
    if isnothing(prob.set_dynamics_cache!)
        g_dynamics_ref = set_dynamics_cache!(prob, x_ref, u_ref)         # default implementation
    else
        g_dynamics_ref = prob.set_dynamics_cache!(prob, x_ref, u_ref)    # user-defined implementation
    end
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )

    # define stacked flattened variables difference
    if prob.ng > 0 || prob.nh > 0
        Δz = stack_flatten_variables(prob, prob.model[:x] - x_ref, prob.model[:u] - u_ref)
    end

    # set nonconvex equality constraints
    if prob.ng > 0
        set_g_noncvx_cache!(prob.lincache, prob.∇g_noncvx, x_ref, u_ref)
        g_ref = prob.g_noncvx(x_ref, u_ref)
        @constraint(prob.model, constraint_g_noncvx[i in 1:prob.ng],
            g_ref[i] + prob.lincache.∇g[i,:]' * Δz == prob.model[:ξ][i]
        )
    else
        g_ref = nothing
    end

    # set nonconvex inequality constraints
    if prob.nh > 0
        set_h_noncvx_cache!(prob.lincache, prob.∇h_noncvx, x_ref, u_ref)
        h_ref = max.(prob.h_noncvx(x_ref, u_ref), 0)
        @constraint(prob.model, constraint_h_noncvx[i in 1:prob.nh],
            h_ref[i] + prob.lincache.∇h[i,:]' * Δz <= prob.model[:ζ][i]
        )
    else
        h_ref = nothing
    end
    return g_dynamics_ref, g_ref, h_ref
end