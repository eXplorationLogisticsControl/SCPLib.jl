# SCPLib.jl — adversarial review of src/ (SCvx*, SCvx, prox-linear, fixed-TRW, discretization)

Repo state reviewed: `main` @ 9ee12ec (plus uncommitted docs changes). No repo files were modified.

> **Independent check (2026-11-03):** C1, C2, M1, M2, M3 and M8 were re-checked against the source. C1 was reproduced by re-running `r01_hclip.jl`: SCvx* with the auto-tuned weight, ProxLinear and SCvx all return `:Optimal` at x₂ ≈ 0 (true optimum 1.0). SCvx* with `w0 = 10` does reach 1.0 after 21 iterations, so the SCvx* failure also depends on M5. Line references for `linearization.jl` were corrected (C2 → line 88, impulsive Φ_B → line 44). The remaining findings have not been independently checked.

> **Fix status (2026-11-03):** C1 and M1 are fixed on branch `debug/scvxstar-inequality-linearization` (commit `1cc087c`). Tests are in `test/test_scvxstar_inequality_linearization.jl`, and the full suite passes (81/81). See the **Resolution** notes under C1 and M1.
>
> **Fix status (2026-11-03, second pass):** M2, M4 and M10 are fixed, M9 is mitigated (opt-in fixed and documented, default unchanged by decision), and M3 is fixed too, because the corrected ratio test needs it (see the M2 note). Tests are in `test/test_scp_globalization.jl`, and the full suite passes (158/158). An adversarial review of this pass found 2 high-severity defects in the first design, which were fixed (see "Adversarial review" under M10). Four regression constants were re-baselined; each one is justified in the **Resolution** notes. All other findings are still open.

Repro scripts (run from the repo root with `julia --project=. <script>`):

- `dev/scp_review_repros/r01_hclip.jl`: false convergence from the clipped inequality linearization
- `dev/scp_review_repros/r02_misc.jl`: forward-backward sign, ξ_dyn decoupling, model growth, ρ with ΔL<0, warm start crash, re-entrancy, μ monotonicity, AD and implicit solver. Cleaned output: `dev/scp_review_repros/r02_misc.clean.out`
- `dev/scp_review_repros/r03_ad.jl`: control case for the AD eom_aug! issue

Notation: z = (x,u), z̄ = reference, z* = subproblem solution, h⁺ = max(0,h).

---

## Ranked summary

| ID | Sev | Status | Location | One-liner |
|---|---|---|---|---|
| C1 | Critical | FIXED | nlconstraints.jl:93 | Inequalities are linearized as `max(h(z̄),0) + ∇h Δz ≤ ζ`, not `h(z̄) + ∇h Δz ≤ ζ`. Inactive constraints become one-sided barriers, which causes false "Optimal" (SCvx*, SCvx, ProxLinear). |
| C2 | Critical (latent) | CONFIRMED | linearization.jl:88 | Forward-backward ∂g/∂x₁ has the wrong sign. The test hides it with a sign-flipped FD and `maximum(err)` without `abs`. |
| M1 | Major (Critical with C1) | FIXED | scvxstar.jl:429,502 | μ is updated with clipped h⁺, so μ never decreases. Complementarity is violated (Oguri uses μ ← max(0, μ + w h)). |
| M2 | Major | FIXED | scvxstar.jl:438; scvx.jl:277; fixedtrw.jl:218 | ρ = ΔJ/ΔL with ΔL<0 and ΔJ<0 gives ρ ≫ 1, so a merit-increasing step is accepted and the TR expands. ΔL≈0 forces ρ=1. |
| M3 | Major | FIXED | nlconstraints.jl:67-70 | Forward-backward: the linearized match-point is a hard equality and ξ_dyn is a free, decoupled variable. Predicted penalty = −‖λ‖²/(2w), which corrupts ρ. Artificial infeasibility is possible. |
| M4 | Major | FIXED | scvx.jl:91-95,70-78; proxlinear.jl:60-91 | SCvx and ProxLinear add new slack variables and cones every iteration and never delete them. The model grows linearly. ProxLinear also builds the ϵ_g slacks twice. |
| M5 | Major | CONFIRMED (value) | scvxstar.jl:242-243, 52 | The auto-tuned w is about 10/χ₀, so w = 1e9 for a feasible guess. w_max = 1e16 by default, which makes the QP terribly conditioned. Contributes to C1 false convergence. |
| M6 | Major | CONFIRMED | utils.jl:76-83; scvxstar.jl:391,492 | `warmstart_dual=true` crashes with KeyError because it sets dual starts on constraints deleted at re-linearization. |
| M7 | Major | CONFIRMED (FB) / LIKELY | scvx.jl:266; proxlinear.jl:268; fixedtrw.jl:207; scvxstar.jl:227 | SCvx, ProxLinear and FixedTRW ignore `fun_get_trajectory` and `shooting_method`. The `tune_initial_penalty_weight!` call crashes for forward-backward. The merit is inconsistent with the custom J_ref. |
| M8 | Major | LIKELY | scvxstar.jl:459; scvx.jl:298; fixedtrw.jl:238 | The returned solution is the last candidate even when it was rejected, so the status describes a rejected point. CPFailed at iteration 1 returns zeros. |
| M9 | Major/Design | MITIGATED | trust_region.jl:46-56; scvxstar.jl:72-76 | Default TR covers only x, not u, tf-dilation or slacks. The TR-convergence argument (ρ→1 as Δ→0) does not hold, so the algorithm can stall or accept with an inaccurate model. |
| M10 | Major/Design | FIXED | scvxstar.jl:472; scvx.jl:314 | The stopping test uses absolute \|ΔJ\| on the candidate, even when rejected or with tiny Δ. It can declare "Optimal" at non-stationary points. |
| M11 | Major/Design | LIKELY | impulsive_problem.jl:64,77; linearization.jl:41-43 | The impulsive jump is `x + dfdu(x,u,t)·u`, and Φ_B = Φ_A·dfdu. This is only correct for jumps that are linear in u and independent of x. |
| M12 | Major/Design | LIKELY | nlconstraints.jl:47-48 vs docs/src/advanced/set_linearized_constraints.md | The docs tell the callback to emit `:constraint_g_noncvx`/`:constraint_h_noncvx`, but the library now emits them too, causing a JuMP name collision. The callback's g_ref/h_ref are ignored. Anonymous callback constraints are never deleted, so stale linearizations accumulate. |
| m1 | Minor | LIKELY | proxlinear.jl:295-306 | ProxLinear breaks before storing history: the final iterate is missing from info, n_iter and the callback, and the exit print shows the previous iterate. |
| m2 | Minor | CONFIRMED | scvxstar.jl:349,365-369; proxlinear.jl:224 | `solve!` is not re-entrant (duplicate `constraint_dynamics` error). `model_nl_references` is mutated (`append!`/`filter!`). |
| m3 | Minor | CONFIRMED | dynamics.jl:46-47 | The AD-generated `eom_aug!` fails with implicit/autodiff ODE solvers (Rodas5) because of the nested-dual output buffer. |
| m4 | Minor | LIKELY | fixedtrw.jl (whole) | FixedTRW has no globalization, uses a pure quadratic penalty (χ bias O(‖∇f‖/w)), and its `@assert tol_feas <= w` is meaningless. |
| m5 | Minor/Design | LIKELY | proxlinear.jl:54-61,102 | Prox-linear has a fixed prox weight that is not tied to w_ep·Lip(∇G). There is no prox on u by default, steps are always accepted, and the stopping test uses only the f₀ change. |
| m6 | Minor | LIKELY | scvxstar.jl:400-408 etc. | `SLOW_PROGRESS` iterates are used. `primal_status` is not checked. Δ reaching Δ_min is not detected (wastes iterations). |
| m7 | Minor | SUSPECT | scvxstar.jl:428-429; :231-232 | g/h at the candidate (and in the w auto-tune) are evaluated with a lincache that holds reference (or zero) data. |
| m8 | Minor | SUSPECT | continuous_problem.jl:146-184 | Forward-backward shares one `_u_k` buffer across all segment ODESolutions, so their lazy interpolants use the wrong control. |
| m9 | Minor | SUSPECT | nlconstraints.jl:12 vs scvxstar.jl:419 | g_dyn_ref comes from the augmented propagation and g from the plain propagation. Different step sequences give ΔJ noise of about w·g·δg, which is non-negligible near w_max. |
| m10 | Minor | LIKELY | linearization.jl:41 | Impulsive Φ_B uses `dfdu(x_ref, u_ref)` without `u_bias`, while the propagation uses u_ref+u_bias. |
| m11 | Minor | LIKELY | scvxstar.jl:462-470 | `n_iter` is incremented only when `store_iterates=true`. |
| m12 | Design | LIKELY | scvxstar.jl:124-132 | The `l1_penalty` option adds √w‖·‖₁ to Oguri's AL, which changes the multiplier fixed point. |
| m13 | Design | LIKELY | impulsive_problem.jl:295 | With Nu=N, the final impulse u[:,N] appears in no dynamics constraint. The user must hand-code it. |
| m14 | Minor | LIKELY | problems/quadcopter.jl:35-38 | `quadcopter_dfdx` gives NaN at v=0. |

---

## Critical

### C1: Inactive non-convex inequalities are linearized around `max(h,0)`, which causes false convergence

- **Severity:** Critical. **Status:** CONFIRMED (`r01_hclip.jl`).
- **Location:** `src/nlconstraints.jl:93-96`
  ```julia
  h_ref = max.(prob.h_noncvx(prob.lincache, x_ref, u_ref), 0)
  @constraint(prob.model, constraint_h_noncvx[i in 1:prob.nh],
      h_ref[i] + prob.lincache.∇h[i,:]' * Δz <= prob.model[:ζ][i])
  ```
- **Theory:**
  - Oguri 2023, subproblem (and Mao et al. SCvx): `h(z̄) + ∇h(z̄)(z−z̄) ≤ ζ, ζ ≥ 0`. The clipping belongs only to the merit (J uses h⁺), not to the model.
  - For prox-linear, the convex model of `max(0,h(z))` is `max(0, h(z̄)+∇h Δz)`. Its directional derivative at a strictly inactive point is 0.
- **Code:** when h(z̄) < 0, the model becomes `0 + ∇hΔz ≤ ζ`. Any motion with ∇h·Δz > 0 is charged μ·(∇hΔz) + w/2(∇hΔz)² in SCvx*, w·∇hΔz in SCvx and w_ep·∇hΔz in ProxLinear. This applies regardless of how far the point is from the boundary. The model is not first-order consistent with the merit, so fixed points of the iteration are not KKT points.
- **Repro:** `min −x₂` s.t. `x₂ ≤ 1` (convex) and `h = x₂ − 2 ≤ 0` (passed as non-convex, never active). Start at x₂ = 0. Output:
  ```
  SCvx*, w0 = nothing   : status=Optimal iters=1 x2=1.8e-9   (true optimum 1.0)  w=1e9
  ProxLinear(1e2,1)     : status=Optimal x2=-9.1e-10
  SCvx (w=1e3)          : status=Optimal iters=1 x2=-2.1e-9
  same h as convex      : x2=1.0 (SCvx*, ProxLinear)
  ```
  The algorithm reports `:Optimal` with χ = 0 at a point that is not stationary. Real cases: the quadcopter obstacle constraints (`R − ‖p − p_obs‖`) penalize every node for approaching an obstacle at a rate of μ or w_ep per unit distance, even when the node is far away. Expect biased, too-conservative trajectories and early stops. The regression test J0 values were generated by this code, so they cannot catch it.
- **Fix:** use the raw `h_ref = h(z̄)` in the constraint (ζ ≥ 0 already gives L(z̄) = J(z̄)). Keep h⁺ only for J_ref, J and χ. For ProxLinear, keep ζ ≥ 0 and use the raw h.
- **Resolution (FIXED, `1cc087c`):**
  - **Change:** `set_linearized_constraints!` now builds `h_raw[i] + ∇h[i,:]'Δz ≤ ζ[i]` and still returns the clipped `h_ref = max.(h_raw, 0)`, which every algorithm uses only for J_ref. ProxLinear, SCvx and FixedTRW needed no changes, because ζ ≥ 0 makes each one's ζ-penalty the correct model of h⁺.
  - **Verification:** a numerical check with h inactive gave L(z̄) − J(z̄) = −5.6e-10.
  - **`r01_hclip.jl`:** every variant now reaches x₂ ≈ 1.0 with `:Optimal`. This includes SCvx* with the auto-tuned w (still about 2e9, see M5), SCvx and ProxLinear.
  - **Tests:** the subproblem RHS encodes the raw h(z̄), and the inactive-constraint problem reaches x₂ = 1 for SCvx* (w0 = 10, auto, `l1_penalty`), SCvx and ProxLinear. On the pre-fix code these fail (except the SCvx* `w0 = 10` case, which also passed before).
  - **Regression value:** `test_scvxstar_h_noncvx.jl` J0 went from 89.5490 to 89.3566. The old solution kept a 0.05 margin from the obstacles; now max h = −8e-11 (constraints active) with dynamics defect 4.7e-10. The test now also asserts max h ≤ 1e-6.
  - **FixedTRW:** it was not affected in practice, because with a quadratic penalty the barrier is only second order. It has no dedicated test.

### C2: Forward-backward Jacobian ∂g/∂x₁ has the wrong sign

- **Severity:** Critical (latent: invisible when x₁ is fixed to x_ref[:,1]). **Status:** CONFIRMED (`r02_misc.jl` (a)).
- **Location:** `src/linearization.jl:88`: `lincache.∇g_dyn[1:nx,1:nx] = prod(reverse(Φ_A_list[1:Nu_fwd]))`
- **Theory:** g = x_bwd(match) − x_fwd(match), so ∂g/∂x₁ = −Φ_{Nu_fwd}⋯Φ₁. The code's own forward-u blocks correctly carry the minus sign (line 308/310).
- **Repro output:**
  ```
  analytic dg/dx1 = [ 0.883  0.480; -0.457  0.884]
  FD       dg/dx1 = [-0.883 -0.480;  0.457 -0.884]
  x_N block and u blocks match FD to 2e-10.
  ```
  For a 0.01 step in x₁: true g = [0.4930, 0.1358]; code model = [0.5010, 0.1090]; correct model = [0.4930, 0.1358].
- **Masking in test:** `test/test_scvxstar_forwardbackward.jl:145` multiplies the FD by −1 ("forward segment has minus sign"). Line 173 uses `@test maximum(err) < 1e-6` without `abs`, which passes for arbitrarily large negative errors.
- **Consequence:** any forward-backward problem with free or partially free initial state (departure on a manifold, free phasing) gets the wrong descent model. ρ becomes negative or random, the TR collapses, and the result is CPFailed or a wrong answer.
- **Fix:** negate line 303. Fix the test to use the raw FD and `maximum(abs.(err))`.

---

## Major

### M1: SCvx* inequality multiplier update uses h⁺, so μ is monotone and complementarity fails

- **Status:** CONFIRMED (`r02_misc.jl` (i)).
- **Location:** `scvxstar.jl:429` (`h_noncvx = max.(h,0)`) is reused at `scvxstar.jl:502` as `algo.μ = algo.μ + algo.w * h_noncvx`.
- **Theory:** Oguri 2023, multiplier update (standard AL for inequalities; Bertsekas): μ ← max(0, μ + w·h(z̄)) with the raw h. μ must be able to decrease to 0 on inactive constraints.
- **Repro:** h = x₂² − 4. Start at x₂ = −3 (violated), optimum x₂ = 1 (inactive). Final state: `status=Optimal, h=-3.0, μ=4.70` (μ history constant after the first update).
- **Consequence:**
  - Returned multipliers are wrong (cannot be used for sensitivity or warm starts).
  - Together with C1, a constraint that was violated once keeps a permanent linear barrier μ·(∇hΔz)⁺ for the rest of the solve.
- **Fix:** keep the raw h for the update, `μ = max.(0, μ .+ w .* h_raw)`, and use h⁺ only in χ and J.
- **Resolution (FIXED, `1cc087c`):** the candidate evaluation keeps both `h_raw` and `h_noncvx = max.(h_raw, 0)`, and the update is `algo.μ = max.(0.0, algo.μ .+ algo.w .* h_raw)`. In the quadcopter test, μ is now 0 on every inactive constraint. Before, 10 inactive constraints had μ > 0.
  - **The repro above, as written (`w0 = 1`), is unchanged by the fix.** SCvx* updates multipliers only on accepted steps with |ΔJ| < δ_i. Here that happens once, at iteration 1 while h is violated, and the run converges before another update, so μ stays at 4.70 because of this gating. This follows Oguri Alg. 1 and is not a remaining bug. With `w0 = 10` the gated update fires at h = −3 and μ goes from 47.03 to 0. The test uses that setting.
  - **Remaining caveat (open):** the returned μ holds the last *gated* update, so it can be stale or non-complementary at convergence. Either apply one final `max(0, μ + w·h)` update on exit or document this.

### M2: Ratio test when ΔL ≤ 0 (and when ΔL ≈ 0)

- **Status:** CONFIRMED (`r02_misc.jl` (f)).
- **Location:** `rho_i = abs(ΔL) > 1e-12 ? ΔJ / ΔL : 1.0` at `scvxstar.jl:438`, `scvx.jl:277` and `fixedtrw.jl:218`.
- **Theory:**
  - SCvx and SCvx* guarantee ΔL ≥ 0 only if z̄ is feasible for the convex subproblem.
  - ΔL ≤ 0 means either stationarity (stop) or a convex-infeasible reference. In both cases it must not be scored as a "very good" step.
  - Mao 2018: stop when ΔL = 0.
- **What happens:**
  - If the initial guess violates the convex constraints (boundary conditions, control bounds), then ΔL < 0. When ΔJ < 0 too, ρ > 0 and the step is accepted. When ΔJ ≪ ΔL, ρ ≫ 1 and the TR expands.
  - Repro: `dx = u + u³`, x₁ = 0 and x₂ = 1 imposed. Output: `ΔJ = −4993, ΔL = −1.0, ρ = 4.99e3, accepted, Δ: 2 → 6`. A step that worsened the merit by 5e3, with a model wrong by 5000×, enlarged the TR.
  - If signs differ (ΔJ > 0 > ΔL), a genuinely improving step is rejected. The reference stays convex-infeasible and Δ shrinks until the subproblem is infeasible (CPFailed).
  - `|ΔL| ≤ 1e-12 ⇒ ρ = 1` (absolute threshold, not scaled to J) accepts and expands on degenerate subproblems where z* ≠ z̄ but L(z*) = L(z̄) (e.g. non-unique LP solutions), whatever ΔJ is.
  - `test/test_scvxstar_callback.jl` relies on the ΔL<0 acceptance path.
- **Fix:**
  - If ΔL ≤ ε·max(1,|J_ref|): stop if χ ≤ tol, otherwise reject and shrink.
  - Treat a convex-infeasible initial reference explicitly: always accept the first step without a ratio test, or project the guess first.
- **Resolution (FIXED):** a new `classify_step` in `trust_region.jl` (used by SCvx* and SCvx) replaces the ρ line. With ε = min(`tol_ΔL`·max(1,|J_ref|), `tol_opt`), where `tol_ΔL = 1e-8` is a new `solve!` kwarg (the cap keeps a constant objective offset from hiding a decrease):
  - **`:restoration`.** The initial guess violates the user's convex constraints. This is checked once, before any linearized or TR constraints exist, with `JuMP.primal_feasibility_report(...; skip_missing=true)`. A `ΔL < −ε` before any accepted step is also taken as a certificate of this. The step is accepted without a ratio test, Δ is kept, and δ_i is not initialised from its ΔJ. Repro (f) now gives `rst`, Δ stays at 2.0 (it was 6.0), and `test_scvxstar_callback.jl` still passes.
  - **`:ratio`.** ΔL > ε uses the usual ρ thresholds.
  - **`:stationary` (null step).** ΔL ≤ ε (of either sign) with the TR inactive at z*. The argument: z̄ is feasible for its own subproblem with L(z̄) = J(z̄), so the subproblem cannot beat it by more than ε. A negative ΔL only means the solver returned something worse than z̄, so z̄ is optimal within solver accuracy. With the TR inactive, z̄ is then stationary for the penalized problem. In that case:
    - if χ(z̄) ≤ tol_feas, return `:Optimal` with z̄;
    - otherwise SCvx* applies the AL update at z̄ (λ, μ = max(0, μ + w h(z̄)), w ← βw) and keeps the reference and the TR. After `max_null_steps = 10` consecutive null steps it returns `:InfeasibleStationary`;
    - SCvx (fixed w) returns the new status `:InfeasibleStationary` (Mao 2018).
  - **`:unresolved`.** |ΔL| ≤ ε with the TR active: the predicted decrease cannot be resolved at this Δ. The reference is kept and the TR **expands**, because shrinking would only make ΔL smaller. At Δ_max this is treated as a null step. ΔL ≈ 0 no longer gives ρ = 1.
  - **`:reject`.** ΔL < −ε with the TR active: the TR shrinks.
  - FixedTRW accepts every step by design. Its displayed ρ is `NaN` when ΔL ≈ 0.
  - **The fix exposed two latent issues that the old acceptance path had masked:**
    - *Forward-backward (`test_scvxstar_forwardbackward.jl`).* It converged only because a ΔL = −0.045 step was scored ρ = 5.95 and accepted. Both x nodes are fixed by the BCs, so the default x-TR constrains nothing (M9). The hard match-point equality (M3) also makes z̄ infeasible for its own subproblem, so L(z̄) ≠ J(z̄). Fixing either one alone gives `:TRCollapsed` (M3 only) or `:CPFailed` (u-TR only). With the M3 fix plus `use_trustregion_control = true` in that test, it reaches `:Optimal` in 23 iterations with ρ ≈ 1 at the end. J0 goes from 1.5812496921 to 1.5812497959 (re-baselined), with χ = 6e-8.
    - *Biased impulsive (`test_scvxstar_impulsive_u_bias.jl`, case with bias).* The old run stopped at iteration 42 on a *rejected* step (ρ = −0.03, M10). The new run reaches w ≈ 2.7e12, where ΔL ≈ −1e-5 is solver noise (M5). The null-step rule above terminates it `:Optimal` with χ ≈ 1e-12. Before ΔL ≤ ε (either sign) with an inactive TR was treated as a null step, it ended `:TRCollapsed`.
  - `test_scvx_impulsive_tr_u.jl`: the old run accepted a ΔL < 0 step with ρ = 152 at iteration 39. The new run ends with a feasible null step (χ = 3.9e-10) at a slightly lower objective. J0 goes from 0.2146711336 to 0.2146596098 (re-baselined).

### M3: Forward-backward: hard match constraint plus a decoupled ξ_dyn

- **Status:** CONFIRMED (`r02_misc.jl` (b)).
- **Location:** `nlconstraints.jl:67-70`: `g_dynamics_ref + ∇g_dyn * Δz == zeros(nx)`. `ξ_dyn` is created (`continuous_problem.jl:390`) but does not appear in this constraint.
- **Theory:**
  - SCvx* needs a virtual-control slack ξ inside the linearized constraint (`g + ∇gΔz = ξ`) both to avoid artificial infeasibility and to make L the model of J.
  - J actually penalizes g(z*) (scvxstar.jl:434), but L penalizes a free variable that the solver sets to ξ = argmin λᵀξ + w/2‖ξ‖² = −λ/w.
- **Repro:** λ_dyn = [3,−5], w = 10 gives `value(ξ_dyn) = [-0.3, 0.5] = -λ/w` and a predicted penalty of −1.7 (should be 0).
- **Consequence:**
  - Once λ_dyn ≠ 0 (after the first multiplier update), ΔL is overstated by ‖λ‖²/(2w) every iteration, so ρ is biased low, causing spurious rejections and TR collapse.
  - The hard equality also makes the subproblem infeasible when Δ is small or the bounds are tight.
- **Fix:** `g_ref + ∇g_dyn Δz == ξ_dyn[:,1]`.
- **Resolution (FIXED, together with M2):** applied as proposed in `nlconstraints.jl`. Without it, the corrected ratio test (M2) cannot converge on forward-backward problems (see the M2 note). Forward-backward also needs `use_trustregion_control = true` when both x nodes are fixed, because the x-TR is then vacuous. `test_scvxstar_forwardbackward.jl` now sets it. C2 (wrong sign of ∂g/∂x₁) is still open; it is masked in the test because x₁ is fixed.

### M4: Model growth in SCvx and ProxLinear

- **Status:** CONFIRMED (`r02_misc.jl` (d), (e)).
- **Location:**
  - `scvx.jl:91-95` creates 1–3 new slack variables per iteration. `scvx.jl:70,73,78` add anonymous NormOneCone constraints per iteration. None are deleted.
  - `proxlinear.jl:60-61,64-67,72-75,82-88` create new ϵ_proximal (SOC), ϵ_noncvx_dyn (nx·(N−1) vars + 3 sets of constraints) and ϵ_g per iteration, none deleted. Lines 70-78 and 80-91 build the ϵ_g slack twice; the first, vector version is dead but stays in the model.
- **Repro:**
  - SCvx: after 6 iterations, 6 NormOneCone constraints exist (5 stale) and nvars grew 4 → 10.
  - ProxLinear: after 50 iterations, 50 SOC cones, 100 extra GreaterThan constraints and nvars 4 → 104.
  - For N=100, nx=6, each ProxLinear iteration adds about 600 variables and 1800 constraints.
- **Consequence:**
  - Solve time grows with the iteration count.
  - The stale epigraph variables are free and unbounded above, so the optimal face is unbounded. Interior-point solvers then see poorer conditioning, ALMOST_OPTIMAL or SLOW_PROGRESS.
- **Fix:** create the slacks once (as SCvxStar does for l1) and register and delete the cone constraints with the other references.
- **Resolution (FIXED):**
  - **SCvx:** `add_l1_slacks!` creates the three epigraph variables and their NormOneCone constraints once per `solve!`. They do not depend on the reference. `penalty(::SCvx, …, slacks_L1)` now only builds the expression.
  - **ProxLinear:** `add_proxlinear_slacks!` creates `ϵ_noncvx_dyn` (sized from `ξ_dyn`, so forward-backward no longer indexes N−1 columns), a single NormOneCone `ϵ_noncvx_g` (the dead vector duplicate is removed) and the `ϵ_proximal` variable once. The reference-dependent SOC prox constraint is deleted and re-added each iteration.
  - Both `solve_convex_subproblem!` methods take the slacks as an optional last argument, so standalone calls still work.
  - **Repros:** (d) has 1 NormOneCone and 5 variables after 6 iterations (was 6 cones and 10 variables); (e) has 1 SOC and 6 variables after 50 iterations (was 50 SOC and 104 variables).
  - **Tests:** the variable and constraint counts are constant across iterations.
  - The SCvx* `l1_penalty` cones are still deleted and re-created every iteration. That is bounded, not growth.

### M5: Penalty weight defaults (auto-tune and w_max)

- **Status:** CONFIRMED (`r01`: auto-tuned w = 1.0e9).
- **Location:** `scvxstar.jl:242-243` (`w = K_w (1+ε)/(χ+ε)` with ε = 1e-8) and `scvxstar.jl:52` (`w_max = 1e16`).
- **Consequence:**
  - A dynamically feasible initial guess (χ ≈ 0, common for propagated guesses) gives w ≈ 1e9 from iteration 1. With β = 2, w can reach 1e16.
  - The QP then has Hessian entries of about 5e15 against O(1) objective terms. IPM tolerances (about 1e-8) on ξ translate into merit errors of order w·ξ·δξ, so the ratio test becomes noise (see also m9).
  - In C1 the auto-tune is what freezes the iterate: step = 1/w = 1e-9.
- **Fix:** clamp the auto-tune (e.g. `w ∈ [w_min, w_max_init]`) and default `w_max` to about 1e8 or less. Consider scaling by |J|.

### M6: `warmstart_dual` crashes

- **Status:** CONFIRMED (`r02_misc.jl` (g)): `KeyError: key ConstraintIndex{VectorAffineFunction, Nonpositives}(1) not found`.
- **Location:**
  - `scvxstar.jl:492` stores `get_constraint_solutions(prob.model)`, which includes `constraint_dynamics`, TR and g/h constraints.
  - The next iteration deletes them (`:367`) and then calls `set_optimal_start_values` on those deleted refs (`:391`, `utils.jl:80-82`).
  - Same pattern in `scvx.jl:239,331`.
- **Fix:** skip invalid refs (`is_valid(model, ci)`), or map duals by constraint name/index after re-creation.

### M7: SCvx, ProxLinear and FixedTRW ignore `fun_get_trajectory` and `shooting_method`; auto-tune breaks forward-backward

- **Status:** CONFIRMED for forward-backward (`r02_misc.jl` (c): `BoundsError ... [1:2, 3]` from `tune_initial_penalty_weight!` with `w0=nothing`); LIKELY for the rest.
- **Location:**
  - `scvx.jl:266`, `proxlinear.jl:268` and `fixedtrw.jl:207` always call `get_trajectory(prob, _x, _u)`.
  - `scvxstar.jl:227` calls multiple-shooting `get_trajectory` in auto-tune even for forward-backward.
- **Consequence:**
  - When the user supplies `set_dynamics_cache!`/`fun_get_trajectory` (custom propagation or lincache layout), J_ref is built from the user's g_dyn_ref while J uses the library's default propagation. The merit is inconsistent, so ρ is meaningless (or it crashes for custom lincache layouts).
  - Forward-backward plus SCvx/ProxLinear/FixedTRW crashes: `ξ_dyn` is nx×1 but `ϵ_noncvx_dyn` is nx×(N−1) in ProxLinear, and `get_trajectory` indexes N columns.
- **Fix:** factor out a single `evaluate_nonlinear_constraints(prob, x, u)` used by every algorithm and by the auto-tune.

### M8: The returned iterate can be a rejected candidate

- **Status:** LIKELY (by reading).
- **Location:** `scvxstar.jl:459-460` (`solution.x[:,:] = _x` before the acceptance test), `scvx.jl:298`, `fixedtrw.jl:238`. Status at maxiter (`scvxstar.jl:517-522`) uses χ of the same candidate.
- **Theory:** SCvx and SCvx* return z̄, the last accepted iterate, whose merit is monotonically non-increasing.
- **Consequences:**
  - If the last iteration is rejected, the user receives a point with a worse merit than x_ref, with status `:Feasible`/`:MaxIterReached` describing that rejected point.
  - On `:CPFailed` at iteration 1, `solution.x`/`u` are zeros (`SCvxStarSolution` constructor).
- **Fix:** copy into `solution` only on accept (or on convergence); on CPFailed return x_ref/u_ref.

### M9: The default trust region does not cover all decision variables

- **Status:** LIKELY (design).
- **Location:** `trust_region.jl:46-56` (x only unless `use_trustregion_control`). The default is `false` (scvxstar.jl:56, scvx.jl:31). Free-final-time examples put the time dilation in u[5] (examples/ex_scvxstar_cr3bp_freetf_ad.jl:39).
- **Theory:** the convergence proofs of SCvx (Mao 2018, Thm. 3) and SCvx* (Oguri 2023) use ‖z − z̄‖ ≤ Δ over all z. That bound is what makes ρ → 1 as Δ → 0.
- **Code:** with x pinned, the segment defect is still nonlinear in u_k (state-dependent B, the bilinear term s·f(x,u) for tf dilation, nonlinear g/h in u). The model error therefore does not vanish as Δ → 0.
- **Consequence:**
  - Repeated rejections drive Δ → Δ_min = 1e-10 with no progress until maxiter.
  - Alternatively, a large unbounded jump in u (or tf) is accepted thanks to M2.
  - Slack variables are naturally excluded (fine), but tf must be included.
- **Fix:** enable the u-TR by default (at least for components that enter nonlinearly), or document that convergence guarantees need it.
- **Resolution (MITIGATED; by decision, the default stays `false` so existing scripts keep their iterates):**
  - The `SCvxStar`/`SCvx` docstrings now document `use_trustregion_control`/`Δ0_u` and explain why the convergence theory needs them.
  - `solve!` emits a one-time `@warn` (verbosity > 0) when the u-TR is off.
  - `use_trustregion_control = true` without `nu` used to log `@error` and then crash in `solve!`. It now builds `tr_u` lazily from `prob.nu` (`ensure_trust_region_control!`), also when `set_trust_region_constraints!` is called directly.
  - u-TR reference names are no longer appended twice.
  - Forward-backward with both x nodes fixed effectively needs the u-TR (see M2/M3).

### M10: The stopping test can fire at non-stationary points

- **Status:** SUSPECT.
- **Location:** `scvxstar.jl:472`, `scvx.jl:314`, `fixedtrw.jl:254`.
- **Code:** `abs(ΔJ) <= tol_opt && χ <= tol_feas` is evaluated on the candidate before the accept/reject decision, with an absolute tolerance.
- **Consequences:**
  - After several rejections Δ is tiny, so |ΔJ| = O(Δ) < tol_opt and a feasible but non-stationary point is declared `:Optimal`.
  - A rejected step with a small merit increase (ΔJ ∈ (−tol, 0)) also terminates as `:Optimal` and returns the rejected point.
  - With C1, ΔJ → 0 because of the barrier, which is exactly how the false convergence shows up.
- **Theory:** Mao's criterion is ΔL ≤ ε (or ‖z*−z̄‖ ≤ ε), and only meaningful when Δ is not at its floor. Oguri's |ΔJ| test likewise presumes Δ bounded away from 0 at acceptance.
- **Fix:**
  - Require the step to be accepted.
  - Use ΔL, or a relative tolerance |ΔJ| ≤ ε(1+|J|).
  - Refuse to declare optimality while Δ is at Δ_min; return `:TRCollapsed` instead.
- **Resolution (FIXED; `tol_opt` stays absolute by decision):**
  - The SCvx*/SCvx test runs after step classification. `:Optimal` requires one of:
    - an accepted ratio step that is **not bounded by the TR** (no component with |z*−z̄| ≥ max((1−1e-3)Δ − 1e-8, Δ/2)), with |ΔJ| ≤ tol_opt and χ ≤ tol_feas;
    - a feasible null step (see M2), which returns z̄;
    - the unchanged `J0 ≤ tol_J0` target.
  - The TR-inactive condition is stronger than "Δ not at Δ_min". It removes the tiny-Δ false stop down to the solver's accuracy: with `Δ0 = 1e-6` the old code returned `:Optimal` at x₂ ≈ 1e-6, and the new code reaches 1. Below Δ ≈ 1e-8, Clarabel's accuracy makes the TR-activity test noise. The initial Δ is therefore clamped into `Δ_bounds` (SCvx has Δ_min = 1e-6). SCvx*'s default Δ_min = 1e-10 sits below that accuracy, but the `:unresolved` → expand rule recovers it (Δ0 = 1e-9 reaches x₂ = 1 in 21 iterations).
  - A rejected step while every Δ (and Δ_u) is already at Δ_min returns the new status `:TRCollapsed` with the last accepted reference.
  - FixedTRW's stop also requires the TR to be inactive.
  - For null steps, the recorded iterate (`solution.x`, `info[:J0]`, `info[:χ]`) is the reference, and `info[:ΔJ]` = 0.
  - **Regression:** `test_scvxstar_dynamics_userdefined.jl` and `test_scvxstar_custom_linearized_constraints.jl` (auto-tuned w) used to stop at iteration 43 on a TR-bounded step (Δ = 1.2e-3). They now stop at iteration 81 with χ = 7e-11. J0 goes from 4.8947284671 to 4.8946695762 (re-baselined); the AD and out-of-place variants of the same problem reach 4.894668. The cost is more iterations on problems where repeated rejections keep the TR small.
  - **Adversarial review** (independent agent, with repro scripts):
    - **H1 (fixed):** ε scaled with |J_ref|, so with objective `1e6 − x₂` it exceeded `tol_opt`. A real decrease of 5e-3 was then a "null step", giving a false `:Optimal` at x₂ = 0.995. Fixed by capping ε at `tol_opt`.
    - **H2 (fixed):** the first design shrank the TR when ΔL ≈ 0 with the TR active. Once Δ·slope < ε, nothing could recover, so it ended `:TRCollapsed` at x₂ = 0 for objective offsets of 1e4/1e6 with Δ0 = 1e-5/1e-3. Fixed by the `:unresolved` → expand rule. Both cases are regression-tested ("objective offset invariance").
    - **Infeasible stationary point, SCvx\* (fixed):** repeated null steps doubled w until Clarabel failed (`:CPFailed` at w = 1.4e12). It now exits `:InfeasibleStationary` after `max_null_steps`.
    - **Bookkeeping (fixed):** on `:TRCollapsed`, `info`/the callback described the rejected candidate. Now the reference is recorded, as for null steps. Dead `rho_i` initialisation removed; the "certifies" wording in the docstring softened.
    - **Forward-backward with default settings (warning only):** fails (`:TRCollapsed`) without `use_trustregion_control = true`; see M2/M3. `solve!` now prints a specific warning, not maxlog-limited. Auto-enabling it was not done, because that would contradict the default-off decision for M9.
    - **Examples now hit maxiter (not changed):** `ex_scvxstar_cr3bp.jl` and `ex_scvxstar_custom_linearized_constraints.jl` went from `:Optimal` in 68 iterations to `:Feasible` at maxiter = 100. Every accepted step is TR-bounded, in an accept/reject/reject cycle while w doubles, so stationarity is not certified. The final point is better (χ 3e-10 vs 6e-7, J0 2.408563 vs 2.408602). Options: raise `maxiter` there, or add a scale-aware measure for TR-bounded steps (e.g. ΔL/‖Δz‖∞).
    - **Deferred:** a rejected step whose candidate lies strictly inside the TR is re-solved identically until Δ < ‖Δz‖∞ (up to about 29 wasted solves). Shrinking straight to Δ ← min(Δ, ‖z*−z̄‖∞)/α₁ would fix it, but it changes every iteration path and therefore every regression baseline, so it is left for a separate change.
  - **Open:** M8 is still open, so on `:MaxIterReached`/`:Feasible` the returned point can still be a rejected candidate.

### M11: The impulsive jump model is only correct for jumps linear in u and independent of x

- **Status:** LIKELY (design).
- **Location:**
  - `impulsive_problem.jl:64` and `:77` define the jump as `x⁺ = x + dfdu(x,u,t)·u`.
  - `linearization.jl:44` sets `Φ_B = Φ_A·dfdu(x_ref,u_ref,t)`.
  - Φ_A ignores ∂(dfdu·u)/∂x.
- **Theory:** for a jump x⁺ = J(x,u): ∂x_{k+1}/∂x_k = Φ·∂J/∂x and ∂x_{k+1}/∂u_k = Φ·∂J/∂u.
- **Consequence:**
  - The parameter is named `dfdu` (a Jacobian), but it is also used as the jump map.
  - For mass depletion (m⁺ = m·exp(−‖Δv‖/c)), Δv in a rotating or attitude frame, or any B(x), both the propagation and the linearization are wrong.
  - Nothing warns the user.
- **Fix:** take a separate `jump(x,u,t)` and differentiate it (by AD) for Φ_A·∂J/∂x and Φ_A·∂J/∂u.

### M12: The custom `set_linearized_constraints!` contract contradicts the docs; stale anonymous constraints

- **Status:** LIKELY.
- **Location:** `nlconstraints.jl:47-48` and `:80-99` (commit 59577eb). Compare `docs/src/advanced/set_linearized_constraints.md`: "If ng>0 or nh>0, the callback must also emit :constraint_g_noncvx and/or :constraint_h_noncvx".
- **Consequences:**
  - A user who follows the docs gets `An object of name constraint_g_noncvx is already attached`.
  - The returned `g_ref`/`h_ref` are silently discarded (`_, _`). If the user builds g/h with a different linearization or cache, J_ref and the model disagree.
  - Any constraint the callback creates under another name, or anonymously, is never deleted (only `model_nl_references` are deleted). Old linearizations pile up and over-constrain later subproblems, which leads to infeasibility or wrong optima.
  - The forward-backward branch is skipped completely when a callback is set.
- **Fix:**
  - Update the docs.
  - Assert that the callback did not register those names.
  - Let the callback return extra refs to delete, or provide a `prob.model_nl_references` push API in the docs.

---

## Minor

- **m1 ProxLinear history (proxlinear.jl:295-309).** The convergence `break` happens before `push!(info[:J0])`, `n_iter += 1` and the callback. On `:Optimal`, `info[:J0][end]` and the printed "Objective" belong to the previous iterate, and `n_iter` is one short. Fix: store first, then test.
- **m2 Re-entrancy (CONFIRMED, r02 (h)).** A second `solve!` on the same `prob` fails with "An object of name constraint_dynamics is already attached". `append!` of TR-u refs (`scvxstar.jl:349`, `scvx.jl:211`) duplicates entries on re-solve. `proxlinear.jl:224-225` permanently `filter!`s refs out of `prob`. Running SCvx* after ProxLinear on the same `prob` leaves stale TR constraints. Fix: build references locally and clean the model at entry.
- **m3 AD eom_aug! with implicit solvers (CONFIRMED, r03).** `ForwardDiff.jacobian(..., zeros(nx), x)` at dynamics.jl:46-47 uses a Float64 output buffer, so it fails when the ODE solver passes Duals (Rodas5 default autodiff). Plain eom! with Rodas5 works; `Rodas5(autodiff=false)` works. Fix: `similar(x)` buffer, or document. It also allocates a JacobianConfig on every RHS call (performance).
- **m4 FixedTRW.**
  - Pure quadratic penalty without multipliers: at the converged point g ≈ −∇f/w, so feasibility relies on `w = 10/tol_feas`, which conditions badly.
  - Fixed Δ with every step accepted: no descent or convergence guarantee, and it can cycle until maxiter.
  - `@assert tol_feas <= algo.w` (fixedtrw.jl:146) checks nothing meaningful.
  - `info[:Δ]` and `info[:w]` are never filled.
- **m5 Prox-linear theory gaps.**
  - The prox-linear method (Lewis & Wright; Drusvyatskiy & Paquette) needs the prox weight t⁻¹ ≥ w_ep·β (β = Lipschitz constant of ∇G) for sufficient decrease. Otherwise you need a line search or step acceptance on F = f₀ + w_ep‖G‖₁ + w_ep·Σh⁺.
  - Here w_prox is fixed and unchecked, every step is accepted, the prox covers x only (u unbounded unless convex bounds exist), and the stopping test uses only the f₀ change (not ‖z⁺−z̄‖·t⁻¹, the prox-gradient mapping).
- **m6 Solver status handling.** `SLOW_PROGRESS` results are used as if optimal. `primal_status` is never checked. `ALMOST_LOCALLY_SOLVED` is not listed. Δ reaching Δ_min (or a no-op shrink still returning `flag_trust_region=true`) is not detected.
- **m7 Stale cache in g/h evaluation.**
  - `prob.g_noncvx(prob.lincache, _x, _u)` at the candidate uses a lincache holding the reference's STMs. If the user's g reads the cache (that is why it is passed), J is not the true nonlinear merit.
  - `tune_initial_penalty_weight!` evaluates g/h before any cache fill (zeros).
- **m8 Forward-backward `_u_k` aliasing (continuous_problem.jl:146-180, 193-227).** One buffer is shared by all `remake`d problems. The returned `sols[k]` keep `p.u` pointing at the last-filled control, so lazy or dense interpolants (Vern7/8) used for plotting re-evaluate f with the wrong control. Internal use (`sol.u[end]`) is fine. Use `make_u_pool` as in multiple shooting.
- **m9 Two propagations for one merit.** g_dyn_ref comes from the augmented ODE (error control includes the STM, so different steps) and g from the plain ODE. The ΔJ noise is about w·‖g‖·δg. It is negligible at w ≈ 1e4 but not near w_max = 1e16 with 1e-12 tolerances. Use the same propagator, or cache g(z*) from the next augmented propagation.
- **m10 u_bias in impulsive Φ_B (linearization.jl:44).** `dfdu(x_ref[:,k], u_ref[:,k], ...)` omits the bias, while the propagation used `u_ref + u_bias` (impulsive_problem.jl:115-116). This is inconsistent whenever dfdu depends on u.
- **m11 `n_iter`.** It counts only when `store_iterates=true` (scvxstar.jl:469 etc.). With `false`, `n_iter = 0` and the exit summary hides results.
- **m12 `l1_penalty` in SCvx*.** It adds √w‖·‖₁ to both J and L (consistent), but it is not part of Oguri's AL. The multiplier fixed point shifts: stationarity becomes ∇f + (λ + √w·sign)ᵀ∇g, so the λ updates no longer estimate the KKT multipliers. Document it as a heuristic.
- **m13 Impulsive Nu = N.** `u[:,N]` never enters `constraint_dynamics`. Tests and examples hand-code `x[4:6,end] + u[1:3,end] == rvf`, which bypasses `dfdu` and `u_bias`. Easy to get wrong.
- **m14 Small items.**
  - quadcopter `dfdx` divides by ‖v‖ (NaN at v=0).
  - `TrustRegions(nu,N,Δ0_u::Matrix)` requires (nu,N) although u has N−1 columns.
  - Vector Δ0 is labelled `:NodeDependent` (it is state-dependent).
  - `Base.show(::SCvxStar)` with `w=nothing` errors in `@printf`.
  - `solve!` overwrites the caller's `x_ref`/`u_ref` in place (including through an `Adjoint`).
  - A forward-backward problem with ng/nh>0 crashes in `stack_flatten_variables` (expects N columns of x).

---

## Things checked and found correct

- **Augmented STM ODE** (dynamics.jl:20-30, 40-51): Φ̇_A = AΦ_A with Φ_A(0)=I, Φ̇_B = AΦ_B + B with Φ_B(0)=0 (ZOH over the segment). Column-major reshape is consistent between the ODE, the initializer and `set_continuous_dynamics_cache!`.
- **Affine term:** Φ_c = x_f − Φ_A x̄ − Φ_B ū. At Δz = 0 the linearized defect equals g_dyn_ref = x̄_{k+1} − φ(x̄_k, ū_k), and both use the same sign convention as `get_trajectory`. Hence L(z̄) = J(z̄) whenever z̄ is convex-feasible.
- **Forward-backward:** the x_N block and all u blocks (forward with minus, backward with plus) match finite differences to 2e-10. `Nu_fwd` is consistent between the cache and the propagation.
- **Free final time via time-dilation control:** the ∂/∂s column is captured by Φ_B, which is correct for ZOH.
- **u_bias (continuous):** applied identically in `get_trajectory`, `get_trajectory_augmented` and Φ_c, so the model and the merit agree.
- **Ensemble threading:** per-segment control buffers (`make_u_pool`) avoid races in `prob_func`. The shared `placeholder` in the impulsive AD eom is only an output buffer (harmless).
- **Node and segment indexing:** N nodes, N−1 segments, and the impulsive Δv applied at the start of segment k before propagation are consistent across propagation, cache and constraints.
- **SCvx* algorithm logic** (apart from M1/M2):
  - The same `penalty` function is used for J_ref, J and L (including the l1 option and SCvx's L1).
  - J_ref is recomputed every iteration with the current weights (callback-safe).
  - Multipliers and w are updated only on accepted steps, with λ ← λ + w·g(z̄) using the pre-increase w, then w ← min(βw, w_max).
  - δ-gating: δ = ∞ initially, then δ ← |ΔJ|, then δ ← γδ.
  - TR thresholds (ρ₀, ρ₁, ρ₂) and update rules match Oguri Alg. 1 and Mao.
  - Rejected steps keep the reference and its linearization and rebuild only the TR constraints.
  - The TR history push is safe because Δ is reassigned rather than mutated.
- **L1 epigraph:** the NormOneCone epigraphs (t ≥ ‖ξ‖₁) are correct, and ζ ≥ 0 makes ‖ζ‖₁ = Σζ.
- **Default ∇g/∇h via ForwardDiff:** the closure capturing `prob` before assignment works (boxed capture); the Jacobian is taken over the full flattened z with the correct ordering.
- **Sanity runs:** on linear dynamics with only convex constraints, SCvx* and ProxLinear converge to the exact optimum (control runs in r01).
