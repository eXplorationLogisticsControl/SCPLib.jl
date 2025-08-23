"""Setting contraints to convex subproblem"""


"""
Set linearized dynamics constraints for continuous problem
"""
function set_linearized_dynamics_constraints!(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref, y_ref)
    set_continuous_dynamics_cache!(prob.lincache, x_ref, u_ref, sols)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )
    return g_dynamics_ref
end


"""
Set linearized dynamics constraints for impulsive problem
"""
function set_linearized_dynamics_constraints!(prob::ImpulsiveProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref, y_ref)
    set_impulsive_dynamics_cache!(prob.lincache, x_ref, u_ref, sols, prob.dfdu)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )
    return g_dynamics_ref
end


"""
Set linearized constraints
"""
function set_linearized_constraints!(prob::OptimalControlProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint}, y_ref::Union{Matrix,Nothing})
    # set dynamics constraints
    g_dynamics_ref = set_linearized_dynamics_constraints!(prob, x_ref, u_ref, y_ref)

    # define stacked flattened variables difference
    if prob.ng > 0 || prob.nh > 0
        _Δy = y_ref isa Nothing ? nothing : prob.model[:y] - y_ref
        Δz = stack_flatten_variables(prob, prob.model[:x] - x_ref, prob.model[:u] - u_ref, _Δy)
    end

    # set nonconvex equality constraints
    if prob.ng > 0
        set_g_noncvx_cache!(prob.lincache, prob.∇g_noncvx, x_ref, u_ref, y_ref)
        g_ref = prob.g_noncvx(x_ref, u_ref, y_ref)
        @constraint(prob.model, constraint_g_noncvx[i in 1:prob.ng],
            g_ref[i] + prob.lincache.∇g[i,:]' * Δz == prob.model[:ξ][i]
        )
    else
        g_ref = nothing
    end

    # set nonconvex inequality constraints
    if prob.nh > 0
        set_h_noncvx_cache!(prob.lincache, prob.∇h_noncvx, x_ref, u_ref, y_ref)
        h_ref = max.(prob.h_noncvx(x_ref, u_ref, y_ref), 0)
        @constraint(prob.model, constraint_h_noncvx[i in 1:prob.nh],
            h_ref[i] + prob.lincache.∇h[i,:]' * Δz <= prob.model[:ζ][i]
        )
    else
        h_ref = nothing
    end
    return g_dynamics_ref, g_ref, h_ref
end