# User-defined linearized constraints

Each SCP iteration rebuilds the linearized dynamics (and, if present, non-convex path constraints) on `prob.model`. The library function `set_linearized_constraints!` does this by default. Both `ContinuousProblem` and `ImpulsiveProblem` accept an optional callback that **replaces** that function entirely.

Use this when the default JuMP product

```math
x_{k+1} - \bigl(\Phi_{A,k}\, x_k + \Phi_{B,k}\, u_k + \Phi_{c,k}\bigr) = \xi^{\mathrm{dyn}}_k
```

is the wrong structure: for example a block-sparse STM, per-scenario scalar rows, or a custom linearization cache. If you only need a custom STM fill and are happy with the dense product above, pass `set_dynamics_cache!` instead and leave `set_linearized_constraints!` unset.

## Constructor

```julia
prob = SCPLib.ContinuousProblem(
    optimizer, eom!, params, objective, times, x_ref, u_ref;
    set_linearized_constraints! = custom_set_linearized_constraints!,
)

# same kwarg on ImpulsiveProblem
prob = SCPLib.ImpulsiveProblem(
    optimizer, eom!, params, objective, times, x_ref, u_ref;
    set_linearized_constraints! = custom_set_linearized_constraints!,
)
```

## Callback signature

```julia
custom_set_linearized_constraints!(prob, x_ref, u_ref) -> (g_dynamics_ref, g_ref, h_ref)
```

| Return | Meaning |
| --- | --- |
| `g_dynamics_ref` | Dynamics defect at the reference, size `(nx, N-1)` for multiple shooting |
| `g_ref` | Non-convex equalities at the reference, or `nothing` if `ng == 0` |
| `h_ref` | Non-convex inequalities at the reference, or `nothing` if `nh == 0` |

The library does **not** call `set_dynamics_cache!` when this callback is set. Fill `prob.lincache` yourself, or call `SCPLib.set_dynamics_cache!(prob, x_ref, u_ref)` from inside the callback.

If `ng > 0` or `nh > 0`, the callback must also emit `:constraint_g_noncvx` and/or `:constraint_h_noncvx`. Those names live in `prob.model_nl_references` and are deleted together with the dynamics constraint.

## JuMP name `:constraint_dynamics`

SCvx*, SCvx, ProxLinear, and FixedTRW delete `prob.model_nl_references` before each accepted re-linearization (`delete_noncvx_referencs!`). That list always includes `:constraint_dynamics`. The callback **must** register a JuMP constraint with that name, otherwise the next iteration cannot delete the previous linearization.

Each `solve!` also starts by removing every constraint that an earlier `solve!` left in `prob.model`, including the names in `prob.model_nl_references`, so the same problem can be solved again (with the same or another algorithm). Constraints that the callback adds without a name are not removed.

`delete_noncvx_referencs!` accepts a single `ConstraintRef` or any iterable container of refs, so both of these work:

```julia
@constraint(prob.model, constraint_dynamics[k in 1:prob.N-1], ...)          # 1-D, library default
@constraint(prob.model, constraint_dynamics[i in 1:S, k in 1:prob.N-1, j in 1:6], ...)  # 3-D
```

## Simple usage (clone the library default)

This is the usual starting point: fill the default STM cache, then emit the same dense product the library would have written. It is valid for both `ContinuousProblem` and `ImpulsiveProblem` with `shooting_method = :multiple` and `ng = nh = 0`.

```julia
custom_set_linearized_constraints! = function (prob, x_ref, u_ref)
    g_dyn = SCPLib.set_dynamics_cache!(prob, x_ref, u_ref)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (
            prob.lincache.Φ_A[:,:,k] * prob.model[:x][:,k] +
            prob.lincache.Φ_B[:,:,k] * prob.model[:u][:,k] +
            prob.lincache.Φ_c[:,k]
        ) == prob.model[:ξ_dyn][:,k]
    )
    return g_dyn, nothing, nothing
end
```

Runnable copies live in:

- `examples/ex_scvxstar_custom_linearized_constraints.jl`
- `examples/ex_scvxstar_impulsive_custom_linearized_constraints.jl`
- `test/test_scvxstar_custom_linearized_constraints.jl`
- `test/test_scvxstar_impulsive_custom_linearized_constraints.jl`

## Structured constraints

When the STM is block-diagonal (e.g. independent scenarios stacked in `x`), a dense `Φ_A[:,:,k] * x[:,k]` in JuMP instantiates many structural zeros. The callback can instead write only the nonzero blocks, still under the name `:constraint_dynamics`:

```julia
custom_set_linearized_constraints! = function (prob, x_ref, u_ref)
    g_dyn = custom_set_dynamics_cache!(prob, x_ref, u_ref)   # fills a custom lincache
    x = prob.model[:x]
    u = prob.model[:u]
    ξ = prob.model[:ξ_dyn]
    Φ_A, Φ_B, Φ_c = prob.lincache.Φ_A, prob.lincache.Φ_B, prob.lincache.Φ_c
    @constraint(
        prob.model,
        constraint_dynamics[i in 1:S, k in 1:prob.N-1, j in 1:nx_block],
        x[offset_x(i) + j, k + 1] - (
            sum(Φ_A[j, p, i, k] * x[offset_x(i) + p, k] for p in 1:nx_block) +
            sum(Φ_B[j, q, i, k] * u[offset_u(i) + q, k] for q in 1:nu_block) +
            Φ_c[j, i, k]
        ) == ξ[offset_x(i) + j, k]
    )
    return g_dyn, nothing, nothing
end
```

`ImpulsiveProblem` also accepts a custom `lincache` object (any `AbstractLinearizationCache`). Pair that with `set_dynamics_cache!` and `fun_get_trajectory` when the default STM layout `Φ_A[:,:,k]` is the wrong size.

## Notes

- Forward–backward shooting uses a single match-point constraint and `prob.lincache.∇g_dyn`, not the per-interval product. A callback for `shooting_method = :forwardbackward` must emit that form.
- The field and the library function share the name `set_linearized_constraints!`. Call the library method as `SCPLib.set_linearized_constraints!(prob, x_ref, u_ref)`; the user override is `prob.set_linearized_constraints!`.
