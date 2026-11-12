# TODO: open findings from the adversarial review of 2026-11-11

This review covered the whole codebase after `main` (4f86dce) was merged into
`cursor/critical-bug-resolution-8f41`. None of the findings was caused by that merge.
IDs in parentheses (M6, m2, …) refer to `dev/scp_adversarial_review.md`. Items marked
**new** were not in that file.

Since that file was written, C2 (the sign of the forward-backward ∂g/∂x₁) has been fixed in
05c519c and has a regression test (`test_forwardbackward_initial_state_jacobian_sign`).

The review re-checked the following and found no errors: the forward-backward Jacobian
(signs, product ordering, the `Nu_fwd` split), the affine term `Φ_c`, flatten/unflatten
ordering, `evaluate_noncvx_gradient` dispatch, model vs. merit consistency, the
multiplier/penalty/δ updates, and the trust-region update logic.

## Major

- [ ] **`warmstart_dual=true` crashes (M6).** `src/utils.jl:89-91`,
  `src/algorithms/scvxstar.jl:483-485, 618-620`, `src/algorithms/scvx.jl:334-336`.
  Constraint references saved at an accepted step are deleted when the model is
  re-linearized, and the next iteration then sets dual start values on them.
  - Repro: integrator, `u_ref = ones`, SCvx* + SCS, `warmstart_dual = true`,
    `tol_opt = -1` → `KeyError: ConstraintIndex{VectorAffineFunction, Nonpositives}(4)`.
  - The warm-start tests (`test_scvxstar_warmstart`, `test_scvx_warmstart`) start at the
    optimum, so every step is rejected and the crash path never runs. Make them start away
    from the optimum.
- [ ] **SCvx, FixedTRW and ProxLinear ignore `fun_get_trajectory` and `shooting_method`
  (M7).** `src/algorithms/scvx.jl:362`, `src/algorithms/fixedtrw.jl:207` and
  `src/algorithms/proxlinear.jl:291` always call the multiple-shooting `get_trajectory`.
  - With a custom propagator, the reference merit uses the user's propagation but the
    candidate merit uses the default one, so the ratio test compares unlike quantities.
  - SCvx with `:forwardbackward` → `BoundsError [1:1, 3]`.
- [ ] **The documented `set_linearized_constraints!` contract causes a name collision
  (M12).** `docs/src/advanced/set_linearized_constraints.md:42` vs.
  `src/nlconstraints.jl:114-134`. The docs tell the callback to register
  `:constraint_g_noncvx`/`:constraint_h_noncvx`, which the library now registers itself.
  Following the docs gives a duplicate-name error, and the returned `g_ref`/`h_ref` are
  silently discarded.

## Medium

- [ ] **`solve!` cannot be called twice on the same problem (m2).**
  `src/algorithms/scvxstar.jl:458-463`: on `it == 1` nothing is deleted, so leftover
  `constraint_dynamics` collides ("An object of name constraint_dynamics is already
  attached"). `src/algorithms/proxlinear.jl:244-245` permanently removes entries from
  `prob.model_nl_references` with `filter!`.
- [ ] **new: the impulsive `eom_aug!` passes the full augmented vectors to the user's
  `eom!`.** `src/dynamics.jl:61, 78` call `eom!(dx_aug, x_aug, p, t)`; the continuous
  version passes `view(dx_aug, 1:nx)`. A vectorised `eom!` such as `dx .= A*x` works in
  `get_trajectory` but fails in `set_dynamics_cache!` with
  `DimensionMismatch (2×2)*(length 6)`. The `ImpulsiveProblem` docstring says `eom!` works
  on an nx-length state.
- [ ] **new: unsupported configurations are only rejected deep inside `solve!`.** Validate
  them in the constructors and in `solve!`:
  - `ImpulsiveProblem(...; shooting_method = :forwardbackward)` is accepted
    (`src/impulsive_problem.jl:211`), then
    `MethodError: set_dynamics_cache_forwardbackward!(::ImpulsiveProblem, …)`.
  - The algorithm and problem `shooting_method` are not checked against each other:
    `SCvxStar` (default `:multiple`) on a forward-backward problem →
    `DimensionMismatch` from `dot(λ_dyn, ξ_dyn)`.
  - Forward-backward with `N = 2`: `src/linearization.jl:122` takes
    `prod(Φ_A_list[2:end])` of an empty list → `MethodError: one(::Type{Matrix{Float64}})`.
    Use an identity matrix for an empty backward segment, or require `N ≥ 3`.
  - An invalid shooting method in the `SCvxStar` constructor raises `UndefVarError`, and
    `test_scvxstar_coverage.jl` currently locks that behaviour in.
- [ ] **The impulsive/mixed control Jacobian ignores `u_bias` and the dependence of `dfdu`
  on x (m10, M11).** `src/linearization.jl:43-44, 63`: `dfdu(x_ref, u_ref)` is evaluated
  without the bias, although propagation uses `u_ref + u_bias`, and the ∂(dfdu·u)/∂x term
  is missing from Φ_A. This only matters when `dfdu` depends on u or x.
- [ ] **new (design trade-off): the first step at the initial guess skips the ratio test.**
  `src/trust_region.jl:363`: `ref_is_initial_guess && ΔL < -ε` returns `:restoration` even
  when `ref_feasible == true`, so a candidate is accepted whatever its merit. The docstring
  argues this deliberately (`ΔL < -ε` at a feasible z̄ indicates infeasibility). But ε is
  only about 1e-8·|J|, while after the first accepted step up to `ε_null` (about 1e-3·|J|)
  is treated as noise and anything worse is rejected.
  - Example: `classify_step(-10.0, -3e-5, 0.2, true, true, false; tol_abs = 1e-4)` →
    `:restoration` (accepted, although the merit got worse by 10).
  - Consider applying the `ε_null` bound or a merit check here, at least when
    `ref_feasible` is true.

## Minor

- [ ] **`n_iter` is only counted when `store_iterates = true` (m11).** In all four
  algorithms, e.g. `src/algorithms/scvxstar.jl:583`; a converged run reports `n_iter = 0`.
- [ ] **ProxLinear does not record its final iterate (m1).**
  `src/algorithms/proxlinear.jl:318-332`: on convergence it breaks before storing the
  iterate in `info`, incrementing `n_iter` or calling the callback.
- [ ] **Forward-backward segments share one control buffer (m8).**
  `src/continuous_problem.jl:148, 195`: every `sols[k].prob.p.u` ends up with the last
  segment's control, so lazy interpolation (Vern7/8, plotting) re-evaluates the dynamics
  with the wrong control. Copy `_u_k` per segment.
- [ ] **new: `times` rejects standard ranges.** `times::Union{Vector,LinRange}` in
  `src/continuous_problem.jl:17`, `src/impulsive_problem.jl:18` and
  `src/mixed_problem.jl:32`. `0.0:0.5:1.0` and `range(0, 1, length = 3)` (`StepRangeLen`)
  fail at construction. Use `AbstractVector{<:Real}`.
- [ ] **new: problems with the `show` methods.**
  - `show(::SCvxStar)` throws when `w === nothing` (the default),
    `src/algorithms/scvxstar.jl:117`.
  - All `Base.show` methods print with `@printf` to stdout and ignore `io`, so
    `sprint(show, SCvx(1, 3))` captures only the first line.
- [ ] **`FixedTRW` `solve!` ignores the `J_expected` keyword.**
  `src/algorithms/fixedtrw.jl:135`.

## Weak tests

- [ ] The warm-start tests never exercise warm-starting (see M6 above).
- [ ] `test/test_mixed_problem.jl:199` accepts `:Solving` as a final status.
- [ ] The J0 regression check in `test/test_scvxstar_impulsive_u_bias.jl:169` is commented
  out.
- [ ] `test/test_scvxstar_impulsive_u_bias.jl:145`: the terminal constraint uses `u[:, N]`
  without `u_bias[:, N]`; the library never applies that column anywhere.
