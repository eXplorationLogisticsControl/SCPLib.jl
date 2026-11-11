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
Set linearized dynamics for a mixed continuous/impulsive problem.
"""
function set_dynamics_cache!(
    prob::MixedProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint},
)
    sols, g_dynamics_ref = get_trajectory_augmented(prob, x_ref, u_ref)
    set_mixed_dynamics_cache!(prob.lincache, x_ref, u_ref, sols, prob)
    return g_dynamics_ref
end


function set_dynamics_cache_forwardbackward!(prob::ContinuousProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
    sols, g_dynamics_ref = get_trajectory_augmented_forwardbackward(prob, x_ref, u_ref)
    set_continuous_dynamics_cache!(prob.lincache, x_ref, u_ref, sols)
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
    if hasproperty(prob, :set_linearized_constraints!) && !isnothing(prob.set_linearized_constraints!)
        g_dynamics_ref, _, _ = prob.set_linearized_constraints!(prob, x_ref, u_ref)
    else
        # set dynamics constraints
        if prob.shooting_method == :multiple
            if isnothing(prob.set_dynamics_cache!)
                g_dynamics_ref = set_dynamics_cache!(prob, x_ref, u_ref)         # default implementation
            else
                g_dynamics_ref = prob.set_dynamics_cache!(prob, x_ref, u_ref)    # user-defined implementation
            end
            if prob isa MixedProblem
                # Column k < N is a dense stack. Build it from VariableRefs so the
                # arrival-node NaNs in model[:u] are never indexed.
                dyn_cons = Any[]
                for k in 1:prob.N-1
                    uk = VariableRef[prob.model[:u][i, k] for i in 1:prob.nu]
                    push!(dyn_cons, @constraint(prob.model,
                        prob.model[:x][:,k+1] - (
                            prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k]
                            + prob.lincache.Φ_B[:,:,k]*uk
                            + prob.lincache.Φ_c[:,k]
                        ) == prob.model[:ξ_dyn][:,k]
                    ))
                end
                prob.model[:constraint_dynamics] = dyn_cons
            else
                @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
                    prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] + prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
                )
            end
        elseif prob.shooting_method == :forwardbackward
            if isnothing(prob.set_dynamics_cache!)
                g_dynamics_ref = set_dynamics_cache_forwardbackward!(prob, x_ref, u_ref)         # default implementation
            else
                g_dynamics_ref = prob.set_dynamics_cache!(prob, x_ref, u_ref)    # user-defined implementation
            end

            # the linearized match-point defect is relaxed by the virtual control ξ_dyn, which is
            # penalized in the merit function: this avoids artificial infeasibility and makes the
            # reference feasible for its own subproblem, so that L(z̄) = J(z̄)
            Δz = vcat(prob.model[:x] - x_ref..., prob.model[:u] - u_ref...)
            @constraint(prob.model, constraint_dynamics,
                g_dynamics_ref + prob.lincache.∇g_dyn * Δz == prob.model[:ξ_dyn][:,1]
            )
        end
    end

    # define stacked flattened variables difference
    if prob.ng > 0 || prob.nh > 0
        Δz = flattened_delta(prob, x_ref, u_ref)
    end

    # set nonconvex equality constraints
    if prob.ng > 0
        set_g_noncvx_cache!(prob.lincache, prob.∇g_noncvx, x_ref, u_ref)
        g_ref = prob.g_noncvx(prob.lincache, x_ref, u_ref)
        @constraint(prob.model, constraint_g_noncvx[i in 1:prob.ng],
            g_ref[i] + prob.lincache.∇g[i,:]' * Δz == prob.model[:ξ][i]
        )
    else
        g_ref = nothing
    end

    # set nonconvex inequality constraints
    if prob.nh > 0
        set_h_noncvx_cache!(prob.lincache, prob.∇h_noncvx, x_ref, u_ref)
        # The model uses the raw h(z̄): h(z̄) + ∇h Δz <= ζ with ζ >= 0, so that
        # ζ = max(0, h(z̄) + ∇h Δz) is the convex model of h⁺. Clipping h(z̄) here
        # would turn inactive constraints into one-sided barriers. The returned
        # h_ref is clipped, as callers only use it in the merit function (h⁺).
        h_raw = prob.h_noncvx(prob.lincache, x_ref, u_ref)
        @constraint(prob.model, constraint_h_noncvx[i in 1:prob.nh],
            h_raw[i] + prob.lincache.∇h[i,:]' * Δz <= prob.model[:ζ][i]
        )
        h_ref = max.(h_raw, 0)
    else
        h_ref = nothing
    end
    return g_dynamics_ref, g_ref, h_ref
end