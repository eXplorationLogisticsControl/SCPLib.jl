# SCvx*

SCvx* (Oguri, 2023) is the default algorithm in `SCPLib.jl`. Each iteration linearizes the
non-convex problem about a reference solution, solves the resulting convex subproblem inside a
trust-region, and decides from the actual and predicted decrease of a merit function whether to
accept the step. Constraint violations are handled with an augmented Lagrangian, whose
multipliers and penalty weight are updated as the iterates approach stationarity. This drives
the iterates towards feasibility without having to choose a large penalty weight up front.

This page also describes the trust-region machinery (step classification, trust-region tests,
termination), which is shared with [SCvx](@ref scvx-algorithm).

```julia
algo = SCPLib.SCvxStar(nx, N; nh = nh, w0 = 1e2)
solution = SCPLib.solve!(algo, prob, x_ref, u_ref; tol_feas = 1e-6, tol_opt = 1e-4)
```

## [Convex subproblem](@id scp-subproblem)

Let $z = (x, u)$ collect the states $x_k$, $k = 1,\ldots,N$, and the controls $u_k$, and let
$\bar{z} = (\bar{x}, \bar{u})$ be the reference. With multiple shooting, the subproblem is

```math
\begin{aligned}
\min_{z,\, \xi_\mathrm{dyn},\, \xi,\, \zeta} \quad& L(z) = \phi(z) + P(\xi_\mathrm{dyn}, \xi, \zeta)
\\ \mathrm{s.t.} \quad& x_{k+1} = \Phi_{A,k} x_k + \Phi_{B,k} u_k + \Phi_{c,k} + \xi_{\mathrm{dyn},k} \quad k = 1,\ldots,N-1
\\& g(\bar{z}) + \nabla g(\bar{z}) (z - \bar{z}) = \xi
\\& h(\bar{z}) + \nabla h(\bar{z}) (z - \bar{z}) \leq \zeta, \quad \zeta \geq 0
\\& z \in \mathcal{Z} \quad \text{(convex constraints added to } \texttt{prob.model} \text{)}
\\& |x_k - \bar{x}_k| \leq \Delta_k, \quad |u_k - \bar{u}_k| \leq \Delta^u_k \quad \text{(trust-region)}
\end{aligned}
```

where:

- the state-transition matrices $\Phi_{A,k}$ and $\Phi_{B,k}$ are evaluated over segment $k$,
  and $\Phi_{c,k}$ makes the linearization exact at the reference;
- the virtual controls $\xi_\mathrm{dyn}$ and $\xi$, and the slack $\zeta$, relax the linearized
  constraints so that the subproblem is always feasible;
- the trust-region is a componentwise box. $\Delta \in \mathbb{R}^{n_x \times N}$ holds one
  radius per state and node, so `Δ0` may be a scalar, a vector (one radius per state) or a matrix
  (one radius per state and node). The box on the controls is only imposed with
  `use_trustregion_control = true`.

The inequalities use the raw $h(\bar{z})$, not $\max(h(\bar{z}), 0)$, so that inactive
constraints do not act as one-sided barriers; $\zeta = \max(0, h(\bar z) + \nabla h(\bar z)(z - \bar z))$
is then the convex model of $h^+$.

## Augmented Lagrangian penalty

SCvx* penalizes the virtual controls and slacks with

```math
P(\xi_\mathrm{dyn}, \xi, \zeta) = \lambda_\mathrm{dyn}^\top \xi_\mathrm{dyn} + \frac{w}{2}\|\xi_\mathrm{dyn}\|_2^2
+ \lambda^\top \xi + \frac{w}{2}\|\xi\|_2^2
+ \mu^\top \zeta + \frac{w}{2}\|\zeta\|_2^2
```

With `l1_penalty = true`, the term $\sqrt{w}\,(\|\xi_\mathrm{dyn}\|_1 + \|\xi\|_1 + \|\zeta\|_1)$ is
added.

If `w0` is not given, the initial weight is tuned from the constraint violation of the initial
guess, $w = K_w (10^{-8} + J_\mathrm{expected}) / (10^{-8} + \chi(\bar{z}))$, where `J_expected`
(default 1) and `K_w` (default 10) are keyword arguments of `solve!`.

## [Merit function and ratio test](@id scp-merit)

The quality of a candidate $z^*$ is measured on the nonlinear problem. The merit function uses
the actual dynamics defects $g_\mathrm{dyn}(z)$ (propagated state minus next node) and the actual
constraint values, with $h^+ = \max(h, 0)$:

```math
J(z) = \phi(z) + P\big(g_\mathrm{dyn}(z),\, g(z),\, h^+(z)\big)
```

The actual and predicted reductions, and their ratio, are

```math
\Delta J = J(\bar{z}) - J(z^*), \qquad \Delta L = J(\bar{z}) - L(z^*), \qquad \rho = \frac{\Delta J}{\Delta L}
```

The constraint violation reported in the log, and used in the tolerances, is

```math
\chi(z) = \max\left(\|g_\mathrm{dyn}(z)\|_\infty,\, \|g(z)\|_\infty,\, \|h^+(z)\|_\infty\right)
```

## [Step classification](@id scp-step-classification)

A plain ratio test breaks down when $\Delta L$ is zero, negative, or below the solver accuracy, so
each step is first classified. With

```math
\varepsilon = \min\big(\texttt{tol\_ΔL} \cdot \max(1, |J(\bar z)|),\ \texttt{tol\_opt}\big),
\qquad
\varepsilon_\mathrm{null} = \max\big(\varepsilon,\ \texttt{tol\_ΔL\_null} \cdot \max(1, |J(\bar z)|),\ \texttt{tol\_opt}\big),
```

the step is one of:

| Kind | Condition | Action | Log |
|:-----|:----------|:-------|:----|
| restoration | the initial guess violates the convex constraints | accept without ratio test; keep $\Delta$ | `rst` |
| ratio | $\Delta L > \varepsilon$ | accept if $\rho \geq \rho_1$; update $\Delta$ from $\rho$ | `yes` / `no` |
| null | trust-region not limiting and $-\varepsilon_\mathrm{null} \leq \Delta L \leq \varepsilon$ | reference is a stationary point of the subproblem | `nul` |
| unresolved | trust-region limiting and $\lvert\Delta L\rvert \leq \varepsilon$ | keep reference; expand $\Delta$ | `exp` |
| reject | otherwise | keep reference; shrink $\Delta$ | `no` |

For a ratio step, the trust-region is shrunk by $\alpha_1$ if $\rho < \rho_2$, expanded by
$\alpha_2$ if $\rho \geq \rho_3$, and kept otherwise, always within `Δ_bounds`. The defaults are
`rhos = (0.0, 0.25, 0.7)` and `alphas = (2.0, 3.0)`. An unresolved step whose trust-region is
already at its ceiling `Δ_bounds[2]` is treated as a null step.

The threshold $\varepsilon$ is relative to the merit, to account for solver accuracy, but never
above `tol_opt`, so a constant offset in the objective cannot hide a decrease. A negative
`tol_opt` leaves $\varepsilon$ uncapped and disables the stationarity-based exits.

## [When is the trust-region limiting?](@id scp-tr-limiting)

Whether the trust-region limits the step decides between a null and an unresolved step, and
whether a small $|\Delta J|$ certifies convergence. A small $|\Delta J|$ means stationarity only
if the trust-region did not stop the step; otherwise it may only reflect a small $\Delta$. Two
tests are combined:

1. **Geometric**: some component of the step lies on the trust-region boundary,
   $|z - \bar{z}| \geq \max\big((1 - 10^{-3})\Delta - 10^{-8},\ \Delta/2\big)$.
2. **Dual**: the multipliers $\nu$ of the trust-region constraints also matter. The optimal
   value of a convex program is a convex function of the right-hand sides of its constraints,
   with subgradient $-\nu$. Raising every radius to its ceiling $\Delta_\mathrm{max} =$ `Δ_bounds[2]`
   can therefore lower the model by at most

   ```math
   L^*(\Delta) - L^*(\Delta_\mathrm{max}) \leq \sum \nu \cdot (\Delta_\mathrm{max} - \Delta)
   ```

   If this bound is at most `tol_tr_dual` (default `tol_opt`), the trust-region does not limit
   the step, even when the step touches the boundary.

The trust-region counts as limiting only if both tests say so. The dual test is skipped (only
the geometric test is used) when the solver returns no duals, or when `tol_tr_dual ≤ 0`. It is
conservative when $\Delta_\mathrm{max}$ is large, since small residual multipliers of an
interior-point solver are multiplied by a large gap.

## Multiplier and penalty updates

On an accepted step with $|\Delta J| < \delta$, SCvx* updates the multipliers at the new reference
and increases the penalty weight:

```math
\lambda_\mathrm{dyn} \leftarrow \lambda_\mathrm{dyn} + w\, g_\mathrm{dyn}, \qquad
\lambda \leftarrow \lambda + w\, g, \qquad
\mu \leftarrow \max(0,\ \mu + w\, h), \qquad
w \leftarrow \min(\beta w,\ w_\mathrm{max})
```

The inequality update uses the raw $h$, so $\mu$ can decrease back to zero. The stationarity
tolerance $\delta$ is set to $|\Delta J|$ the first time, and multiplied by $\gamma$ afterwards
(restoration steps do not change it). The defaults are `beta = 2.0`, `gamma = 0.9` and
`w_max = 1e16`.

At a null step where the reference is infeasible ($\chi(\bar z) >$ `tol_feas`), the subproblem
cannot move away from the reference with the current multipliers. SCvx* then applies the same
updates at the reference, without moving it. After `max_null_steps` consecutive null steps
(default 10), it exits with `:InfeasibleStationary`.

## [Termination](@id scp-termination)

| Status | Condition |
|:-------|:----------|
| `:Optimal` | one of the tests below |
| `:Feasible` / `:Stalled` | a step is rejected with the trust-region at its lower bound `Δ_bounds[1]`; the last accepted reference is returned, `:Feasible` if $\chi \leq$ `tol_feas` |
| `:InfeasibleStationary` | `max_null_steps` consecutive null steps at an infeasible reference |
| `:Feasible` / `:MaxIterReached` | `maxiter` reached, `:Feasible` if $\chi \leq$ `tol_feas` |
| `:CPFailed` | the convex solver did not return an optimal solution |

`:Optimal` is returned by the first test that passes; `solution.info[:converged_by]` records which:

| `converged_by` | Test |
|:---------------|:-----|
| `:ratio` | accepted ratio step, trust-region not limiting, $\lvert\Delta J\rvert \leq$ `tol_opt`, $\chi \leq$ `tol_feas` |
| `:null` | null step at a reference with $\chi \leq$ `tol_feas` |
| `:tol_J0` | objective $\phi \leq$ `tol_J0` with $\chi \leq$ `tol_feas` |
| `:small_step` | only with `small_step_convergence = true`: accepted ratio step bounded by the trust-region, with $\lvert\Delta J\rvert \leq$ `tol_opt`, $\Delta L \leq$ `tol_opt` and $\chi \leq$ `tol_feas` |

The returned solution is always the current iterate: the candidate of an accepted step,
otherwise the reference. A rejected candidate is never returned.

!!! warning "Small-step convergence"
    `:small_step` is not a stationarity certificate: it also passes when the trust-region has
    shrunk. Use it only when the trust-region is deliberately kept small, so that it stays
    limiting at every iteration and neither `:ratio` nor `:null` can fire. An example is a
    frozen radius on a time-dilation control, as in the
    [hypersensitive example](../examples/example_hypersensitive.md).

## Reading the iteration log

```
Iter |     J0     |    ΔJ_i    |    ΔL_i    |     χ_i    |    ρ_i    |    r_i    |     w     |  acpt. |
  74 |  3.378e+00 |  4.221e-05 |  1.574e-04 |  6.627e-07 |  2.68e-01 |  4.88e-05 |  1.31e+07 |  yes   |
```

`J0` is the objective $\phi$ of the candidate, `r_i` is the first trust-region radius
$\Delta_{1,1}$, and `acpt.` is the step kind from the [step classification](@ref scp-step-classification).
`solution.info` stores the history of the recorded iterate in `:J0`, `:ΔJ`, `:χ`, `:w`, `:Δ` and
`:accept`, together with CPU times in `:cpu_times`.

## Practical guidance

- **Trust-region on the controls.** The convergence theory assumes a trust-region on all
  decision variables. Set `use_trustregion_control = true` (with `nu` and `Δ0_u`) when the
  dynamics or the non-convex constraints are nonlinear in $u$, e.g. a time-dilation control for a
  free final time. Without it, the algorithm may reject step after step, or accept poorly
  modelled steps in $u$.
- **Scaling the radii.** Pass `Δ0` and `Δ0_u` as matrices when variables or nodes have very
  different scales, e.g. radii proportional to the local mesh width.
- **Lower bound.** Keep `Δ_bounds[1]` above the solver accuracy (about `1e-8` for interior-point
  solvers). Below it, the subproblem solution is dominated by solver noise and the last
  iterations before a collapse are spent on noise.
- **Penalty weight growth.** Each accepted step with $|\Delta J| < \delta$ multiplies $w$ by
  $\beta$. If $w$ becomes very large while $\chi$ is already small, the subproblem becomes
  ill-conditioned and $\Delta L$ no longer vanishes with $\Delta$; lower `beta` or `w_max`.
- **Warm starting.** `warmstart_primal` and `warmstart_dual` pass the last accepted solution to
  the solver as a starting point.
- **Shooting method.** `shooting_method = :forwardbackward` in the constructor replaces the
  $N-1$ defect constraints with a single match-point defect.

The full list of arguments is in the docstrings of `SCvxStar` and `solve!` on the
[API: Algorithms](@ref) page.

## References

- K. Oguri, "Successive Convexification with Feasibility Guarantee via Augmented Lagrangian for
  Non-Convex Optimal Control Problems," *2023 62nd IEEE Conference on Decision and Control (CDC)*,
  2023.
- Y. Mao, M. Szmuk, X. Xu, and B. Açıkmeşe, "Successive Convexification: A Superlinearly
  Convergent Algorithm for Non-convex Optimal Control Problems," arXiv:1804.06539, 2018.
