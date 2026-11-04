"""
Tests for the trust-region globalization of SCvx* / SCvx / ProxLinear
(adversarial review items M2, M4, M9, M10 in dev/scp_adversarial_review.md):

- M2: ratio test when ΔL ≤ 0 or ΔL ≈ 0 (`classify_step`, restoration step)
- M4: SCvx / ProxLinear slack variables and cones are created once
- M9: control trust region can be enabled without passing `nu`
- M10: stopping test requires an accepted step that is not bounded by the trust region;
       stall exit at the trust-region lower bound returns the reference; `:InfeasibleStationary`
"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end
if !@isdefined verbosity
    verbosity = 0
end


_glob_lin!(dx, x, pu, t) = (dx[1] = pu.u[1]; nothing)               # dx = u
_glob_cub!(dx, x, pu, t) = (dx[1] = pu.u[1] + pu.u[1]^3; nothing)   # dx = u + u³
_glob_sq!(dx, x, pu, t)  = (dx[1] = pu.u[1]^2; nothing)             # dx = u²


"""1-state toy: x₁ = x1_fix, optional x₂ == x2_eq or x₂ ≤ x2_ub, objective `objective`."""
function _glob_problem(eom!, objective; x_ref = zeros(1, 2), u_ref = zeros(1, 1),
                       x2_ub = nothing, x2_eq = nothing, kwargs...)
    x_ref = copy(x_ref)
    u_ref = copy(u_ref)
    prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, eom!, nothing, objective,
                                    [0.0, 1.0], x_ref, u_ref; kwargs...)
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][1, 1] == 0.0)
    if !isnothing(x2_ub)
        @constraint(prob.model, prob.model[:x][1, 2] <= x2_ub)
    end
    if !isnothing(x2_eq)
        @constraint(prob.model, prob.model[:x][1, 2] == x2_eq)
    end
    return prob, x_ref, u_ref
end

_glob_lin_problem(; kwargs...) = _glob_problem(_glob_lin!, (x, u) -> -x[1, 2]; x2_ub = 1.0, kwargs...)

"""
Custom dynamics cache with the sign of Φ_B flipped: the linear model predicts the wrong
direction for u, so every step that moves u is rejected.
"""
function _glob_wrong_sign_cache!(prob, x_ref, u_ref)
    g_dyn = SCPLib.set_dynamics_cache!(prob, x_ref, u_ref)
    for k in 1:prob.N-1
        prob.lincache.Φ_c[:, k] .+= 2 * prob.lincache.Φ_B[:, :, k] * u_ref[:, k]
        prob.lincache.Φ_B[:, :, k] .*= -1
    end
    return g_dyn
end

_glob_ncons(model) = sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model))


function test_classify_step()
    tol = 1e-8
    # reference not convex-feasible → restoration, regardless of the sign of ΔL
    @test SCPLib.classify_step(-4993.0, -1.0, 0.0, false, false, true; tol_ΔL = tol)[2] == :restoration
    @test SCPLib.classify_step(1.0, 2.0, 0.0, false, false, true; tol_ΔL = tol)[2] == :restoration
    # ΔL < 0 on the first iteration certifies a convex-infeasible reference
    @test SCPLib.classify_step(-4993.0, -1.0, 0.0, true, true, true; tol_ΔL = tol)[2] == :restoration
    # ... but later (reference is a subproblem solution) it means an inconsistent model → reject
    @test SCPLib.classify_step(-4993.0, -1.0, 0.0, true, false, true; tol_ΔL = tol)[2] == :reject
    # regular ratio test
    rho, kind = SCPLib.classify_step(0.5, 1.0, 10.0, true, false, true; tol_ΔL = tol)
    @test kind == :ratio
    @test rho ≈ 0.5
    # ΔL ≈ 0 (relative to |J_ref|) must no longer be scored as ρ = 1
    @test SCPLib.classify_step(-3.0, 1e-9, 1.0, true, false, false; tol_ΔL = tol)[2] == :stationary
    @test SCPLib.classify_step(-3.0, 5e-7, 1e2, true, false, false; tol_ΔL = tol)[2] == :stationary
    # ΔL ≈ 0 while the step is bounded by the TR: unresolved at this Δ → expand, never shrink
    @test SCPLib.classify_step(-3.0, 1e-9, 1.0, true, false, true; tol_ΔL = tol)[2] == :unresolved
    # the threshold is capped by tol_abs, so a constant objective offset cannot hide a decrease
    @test SCPLib.classify_step(5e-3, 5e-3, 1e6, true, false, false; tol_ΔL = tol, tol_abs = 1e-4)[2] == :ratio
    # a non-positive tol_abs (tol_opt = -1 disables convergence) must not make ε negative
    @test SCPLib.classify_step(0.0, -1e-30, 0.0, true, true, false; tol_ΔL = tol, tol_abs = -1.0)[2] == :stationary
    @test SCPLib.classify_step(5e-3, 5e-3, 1e6, true, false, false; tol_ΔL = tol)[2] == :stationary
    # ΔL < 0 from a convex-feasible reference with the TR inactive: the solver cannot improve on
    # the reference (solver noise), so the reference is stationary
    @test SCPLib.classify_step(-1e-5, -1e-5, 0.2, true, false, false; tol_ΔL = tol)[2] == :stationary
end


function test_restoration_step()
    # repro (f): x₂ = 1 is imposed but the initial guess has x₂ = 0 (convex-infeasible)
    prob, x_ref, u_ref = _glob_problem(_glob_cub!, (x, u) -> u[1, 1]^2; x2_eq = 1.0)
    @test SCPLib.reference_convex_feasible(prob, x_ref, u_ref; atol = 1e-6) == false
    algo = SCPLib.SCvxStar(1, 2; w0 = 1e4, Δ0 = 2.0)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 1)
    @test sol.info[:accept] == [true]
    @test all(algo.tr.Δ .== 2.0)              # was expanded to 6.0 by ρ ≈ 5e3 before the fix
    @test x_ref[1, 2] ≈ 1.0 atol = 1e-6         # reference moved to the convex-feasible candidate

    # a convex-feasible guess is recognised as such
    prob, x_ref, u_ref = _glob_lin_problem()
    @test SCPLib.reference_convex_feasible(prob, x_ref, u_ref; atol = 1e-6) == true
end


function test_tr_collapse()   # stall exit at the trust-region lower bound
    for make_algo in (
        () -> SCPLib.SCvx(1, 2; w = 1e3, Δ_bounds = (1e-4, 10.0)),
        () -> SCPLib.SCvxStar(1, 2; w0 = 1e2, l1_penalty = true, Δ_bounds = (1e-4, 10.0)),
    )
        prob, x_ref, u_ref = _glob_lin_problem(set_dynamics_cache! = _glob_wrong_sign_cache!)
        algo = make_algo()
        sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
        @test sol.status == :Feasible                # stall exit; the reference (zeros) has χ = 0
        @test sol.n_iter < 100
        @test !any(sol.info[:accept])
        @test all(algo.tr.Δ .== 1e-4)
        @test sol.x == zeros(1, 2)                  # the reference, not a rejected candidate
        @test sol.u == zeros(1, 1)
    end
end


function test_stationary_start()
    # start at the optimum: the subproblem returns z̄ (ΔL ≈ 0) and z̄ is feasible → Optimal
    for make_algo in (() -> SCPLib.SCvxStar(1, 2; w0 = 10.0), () -> SCPLib.SCvx(1, 2; w = 1e3))
        prob, x_ref, u_ref = _glob_lin_problem(x_ref = [0.0 1.0], u_ref = ones(1, 1))
        sol = SCPLib.solve!(make_algo(), prob, x_ref, u_ref; verbosity = verbosity, maxiter = 10)
        @test sol.status == :Optimal
        @test sol.n_iter == 1
        @test sol.x ≈ [0.0 1.0] atol = 1e-6
    end

    # dx = u², x₂ = -1: no u can satisfy the dynamics and ∂g/∂u = 0 at u = 0, so u = 0 is a
    # stationary point of the penalized problem that is infeasible.
    prob, x_ref, u_ref = _glob_problem(_glob_sq!, (x, u) -> u[1, 1]^2;
                                       x_ref = [0.0 -1.0], x2_eq = -1.0)
    sol = SCPLib.solve!(SCPLib.SCvx(1, 2; w = 1e3), prob, x_ref, u_ref; verbosity = verbosity, maxiter = 50)
    @test sol.status == :InfeasibleStationary
    @test sol.n_iter == 1

    # SCvx* updates the multipliers / penalty at the null step instead of shrinking the TR
    prob, x_ref, u_ref = _glob_problem(_glob_sq!, (x, u) -> u[1, 1]^2;
                                       x_ref = [0.0 -1.0], x2_eq = -1.0)
    algo = SCPLib.SCvxStar(1, 2; w0 = 10.0, beta = 2.0)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 3)
    @test sol.status == :MaxIterReached
    @test !any(sol.info[:accept])
    @test sol.info[:w] == [10.0, 20.0, 40.0]
    @test all(algo.tr.Δ .== 0.05)                 # TR untouched by null steps
    @test only(algo.λ_dyn) ≈ -(10.0 + 20.0 + 40.0) atol = 1e-6

    # ... and gives up with :InfeasibleStationary after `max_null_steps` null steps
    prob, x_ref, u_ref = _glob_problem(_glob_sq!, (x, u) -> u[1, 1]^2;
                                       x_ref = [0.0 -1.0], x2_eq = -1.0)
    sol = SCPLib.solve!(SCPLib.SCvxStar(1, 2; w0 = 10.0), prob, x_ref, u_ref;
                        verbosity = verbosity, maxiter = 100, max_null_steps = 5)
    @test sol.status == :InfeasibleStationary
    @test sol.n_iter == 5
end


function test_objective_offset()
    # A constant offset in the objective must not change the outcome (adversarial review H1/H2):
    # the "ΔL ≈ 0" threshold is capped by tol_opt, and an unresolved step expands the TR.
    for (x2_0, Δ0) in ((0.995, 0.05), (0.0, 1e-5), (0.0, 1e-3))
        for make_algo in (() -> SCPLib.SCvxStar(1, 2; w0 = 10.0, Δ0 = Δ0),
                          () -> SCPLib.SCvx(1, 2; w = 1e3, Δ0 = Δ0))
            prob, x_ref, u_ref = _glob_problem(_glob_lin!, (x, u) -> 1e6 - x[1, 2];
                                               x_ref = [0.0 x2_0], u_ref = fill(x2_0, 1, 1), x2_ub = 1.0)
            sol = SCPLib.solve!(make_algo(), prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
            @test sol.status == :Optimal
            @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
        end
    end
end


function test_no_false_convergence_tiny_tr()
    # Before the fix: iteration 1 moves x₂ by Δ0 = 1e-6, |ΔJ| ≈ 1e-6 < tol_opt and χ = 0,
    # so :Optimal was returned at x₂ ≈ 1e-6 (true optimum x₂ = 1).
    for make_algo in (() -> SCPLib.SCvxStar(1, 2; w0 = 10.0, Δ0 = 1e-6),
                      () -> SCPLib.SCvx(1, 2; w = 1e3, Δ0 = 1e-6))
        prob, x_ref, u_ref = _glob_lin_problem()
        sol = SCPLib.solve!(make_algo(), prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
        @test sol.status == :Optimal
        @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
    end

    # Δ0 below Δ_min (SCvx: 1e-6) is clamped to Δ_min; at Δ ≈ 1e-8 the subproblem solution is
    # below solver accuracy and the TR tests are noise (this used to return :Optimal at x₂ = 0)
    prob, x_ref, u_ref = _glob_lin_problem()
    algo = SCPLib.SCvx(1, 2; w = 1e3, Δ0 = 1e-9)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
    @test sol.status == :Optimal
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
end


function test_model_size_constant()
    # SCvx: repro (d)
    prob, x_ref, u_ref = _glob_lin_problem()
    nvars, ncons = Int[], Int[]
    cb = (a, s, i, J, c) -> (push!(nvars, num_variables(prob.model)); push!(ncons, _glob_ncons(prob.model)))
    sol = SCPLib.solve!(SCPLib.SCvx(1, 2; w = 1e3, Δ0 = 0.01), prob, x_ref, u_ref;
                        verbosity = verbosity, maxiter = 30, callback = cb)
    @test length(nvars) >= 3
    @test all(nvars .== nvars[1])
    @test all(ncons .== ncons[1])
    @test num_constraints(prob.model, Vector{VariableRef}, MOI.NormOneCone) == 1

    # ProxLinear: repro (e)
    prob, x_ref, u_ref = _glob_lin_problem()
    nvars, ncons = Int[], Int[]
    cb = (a, s, i, J, c) -> (push!(nvars, num_variables(prob.model)); push!(ncons, _glob_ncons(prob.model)))
    nvars0 = num_variables(prob.model)
    sol = SCPLib.solve!(SCPLib.ProxLinear(1e2, 1e2), prob, x_ref, u_ref;
                        verbosity = verbosity, maxiter = 50, callback = cb)
    @test length(nvars) >= 3
    @test all(nvars .== nvars[1])
    @test all(ncons .== ncons[1])
    @test nvars[1] == nvars0 + 1 + prob.nx * (prob.N - 1)   # ϵ_proximal + ϵ_noncvx_dyn
    n_soc = sum(num_constraints(prob.model, F, MOI.SecondOrderCone)
                for (F, S) in list_of_constraint_types(prob.model) if S == MOI.SecondOrderCone)
    @test n_soc == 1
end


function test_trust_region_control_without_nu()
    prob, x_ref, u_ref = _glob_lin_problem()
    algo = SCPLib.SCvxStar(1, 2; w0 = 10.0, use_trustregion_control = true, Δ0_u = 0.2)
    @test isnothing(algo.tr_u)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
    @test sol.status == :Optimal
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
    @test size(algo.tr_u.Δ, 1) == 1

    prob, x_ref, u_ref = _glob_lin_problem()
    algo = SCPLib.SCvx(1, 2; w = 1e3, use_trustregion_control = true)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 100)
    @test sol.status == :Optimal
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
end


@testset "SCP globalization" begin
    @testset "classify_step" begin test_classify_step() end
    @testset "restoration step (M2)" begin test_restoration_step() end
    @testset "TR collapse (M10)" begin test_tr_collapse() end
    @testset "stationary reference (M2/M10)" begin test_stationary_start() end
    @testset "no false convergence at tiny Δ (M10)" begin test_no_false_convergence_tiny_tr() end
    @testset "objective offset invariance" begin test_objective_offset() end
    @testset "model size constant (M4)" begin test_model_size_constant() end
    @testset "u trust region without nu (M9)" begin test_trust_region_control_without_nu() end
end
