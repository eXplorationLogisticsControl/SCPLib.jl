# [SCvx](@id scvx-algorithm)

SCvx (Mao et al., 2018) solves the same [convex subproblem](@ref scp-subproblem) as SCvx*, but
penalizes the virtual controls with an exact $\ell_1$ penalty of fixed weight instead of an
augmented Lagrangian. It has no multipliers to update and no weight schedule. In exchange, the
weight $w$ must be large enough for the penalized problem to have the same solutions as the
constrained one.

```julia
algo = SCPLib.SCvx(nx, N; w = 1e3)
solution = SCPLib.solve!(algo, prob, x_ref, u_ref; tol_feas = 1e-6, tol_opt = 1e-6)
```

## Exact penalty

```math
P(\xi_\mathrm{dyn}, \xi, \zeta) = w \left(\|\xi_\mathrm{dyn}\|_1 + \|\xi\|_1 + \|\zeta\|_1\right)
```

In the subproblem, each norm is modelled with an epigraph variable $t \geq \|\cdot\|_1$
(`MOI.NormOneCone`). These variables do not depend on the reference, so they are created once per
solve; only the objective is rebuilt at each iteration. For $w$ above a threshold set by the
Lagrange multipliers of the original problem, a feasible stationary point of the penalized
problem is a stationary point of the original problem.

## What is shared with SCvx*

The following parts are identical to SCvx* and are described on the [SCvx*](scvxstar.md) page:

- the [merit function and ratio test](@ref scp-merit), with this penalty $P$;
- the [step classification](@ref scp-step-classification) into restoration, ratio, null,
  unresolved and rejected steps, with the same thresholds `tol_ΔL` and `tol_ΔL_null`;
- the [trust-region tests](@ref scp-tr-limiting), including the dual test (`tol_tr_dual`);
- the [termination tests](@ref scp-termination), including `solution.info[:converged_by]` and
  the opt-in `small_step_convergence`.

## Differences from SCvx*

| | SCvx | SCvx* |
|:-|:-----|:------|
| Penalty | $w\,\lVert\cdot\rVert_1$, fixed $w$ | augmented Lagrangian, multipliers and $w$ updated |
| Infeasible null step | exits with `:InfeasibleStationary` at once | updates multipliers and $w$; exits after `max_null_steps` |
| Initial weight | `w` (default `1e3`) | `w0`, or tuned from the initial guess |
| Default `Δ_bounds` | `(1e-6, 1e4)` | `(1e-10, 10.0)` |
| Default `tol_opt` | `1e-6` | `1e-4` |
| Shooting | multiple shooting | multiple or forward-backward |

An infeasible null step means the reference is a stationary point of the penalized problem that
violates the constraints. With a fixed weight, no further progress is possible, so SCvx stops and
suggests increasing `w`.

!!! note "Large `Δ_max` and the dual test"
    The default ceiling `Δ_bounds[2] = 1e4` is large, so the dual bound
    $\sum \nu (\Delta_\mathrm{max} - \Delta)$ magnifies small residual multipliers. In practice,
    the dual test then rarely overrides the geometric one. Lower `Δ_bounds[2]` to a radius the
    algorithm can actually use if you want the dual test to take effect.

## Choosing the weight

- Too small a $w$ leads to an infeasible stationary point (`:InfeasibleStationary`), or to
  solutions that trade constraint violation for objective.
- Too large a $w$ makes the merit function dominated by the penalty. Steps then mostly reduce
  the violation, and the objective converges slowly.
- The weight must exceed the magnitude of the Lagrange multipliers of the original problem,
  which are usually not known in advance. Start from the default and increase it if the solver exits with
  `:InfeasibleStationary`.

If choosing $w$ is difficult, use [SCvx*](scvxstar.md), which adapts the weight.

The full list of arguments is in the docstrings of `SCvx` and `solve!` on the
[API: Algorithms](@ref) page.

## Reference

- Y. Mao, M. Szmuk, X. Xu, and B. Açıkmeşe, "Successive Convexification: A Superlinearly
  Convergent Algorithm for Non-convex Optimal Control Problems," arXiv:1804.06539, 2018.
