"""Exercise the SCvx* branches that the physical test problems never reach"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using SCS

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


scvxstar_coverage_g_noncvx(cache, x, u) = [x[1,2]^2 - 0.25]
scvxstar_coverage_h_noncvx(cache, x, u) = [x[1,2]^2 - 1.0]


"""Stand-in for the default multiple-shooting propagation"""
function scvxstar_coverage_get_trajectory(prob, x, u)
    return SCPLib.get_trajectory(prob, x, u)
end


"""
User override of `set_linearized_constraints!`.

Registering the dynamics constraint here bypasses the `shooting_method` dispatch
in `nlconstraints.jl`, which lets a test reach the invalid-shooting-method guard
further down in `solve!`.
"""
function scvxstar_coverage_set_linearized_constraints!(prob, x_ref, u_ref)
    g_dynamics_ref = SCPLib.set_dynamics_cache!(prob, x_ref, u_ref)
    @constraint(prob.model, constraint_dynamics[k in 1:prob.N-1],
        prob.model[:x][:,k+1] - (prob.lincache.Φ_A[:,:,k]*prob.model[:x][:,k] +
            prob.lincache.Φ_B[:,:,k]*prob.model[:u][:,k] + prob.lincache.Φ_c[:,k]) == prob.model[:ξ_dyn][:,k]
    )
    return g_dynamics_ref, nothing, nothing
end


"""Single-state integrator `ẋ = u` steered from 0 to 1"""
function make_scvxstar_coverage_problem(optimizer;
    ng::Int = 0, g_noncvx = nothing,
    nh::Int = 0, h_noncvx = nothing,
    fun_get_trajectory = nothing,
    set_linearized_constraints! = nothing,
    infeasible::Bool = false,
)
    nx, nu, N = 1, 1, 3
    x_ref = [0.0 0.5 1.0]           # already satisfies the boundary conditions
    u_ref = zeros(nu, N-1)

    function eom!(dx, x, pu, t)
        dx[1] = pu.u[1]
        return
    end

    prob = SCPLib.ContinuousProblem(
        optimizer,
        eom!,
        nothing,
        (x, u) -> sum(u.^2),
        LinRange(0.0, 1.0, N),
        x_ref,
        u_ref;
        ng = ng,
        g_noncvx = g_noncvx,
        nh = nh,
        h_noncvx = h_noncvx,
        fun_get_trajectory = fun_get_trajectory,
        set_linearized_constraints! = set_linearized_constraints!,
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,end] == 1.0)
    if infeasible
        @constraint(prob.model, prob.model[:x][1,1] == 1.0)
    end
    return prob, x_ref, u_ref
end


function test_scvxstar_invalid_shooting_method_constructor()
    @test_throws UndefVarError SCPLib.SCvxStar(1, 3; shooting_method = :bogus)
end


function test_scvxstar_trustregion_control_without_nu()
    algo = @test_logs (:error,) match_mode=:any SCPLib.SCvxStar(1, 3; use_trustregion_control = true)
    @test isnothing(algo.tr_u)
end


function test_scvxstar_tune_initial_penalty_weight()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer;
        ng = 1, g_noncvx = scvxstar_coverage_g_noncvx,
        nh = 1, h_noncvx = scvxstar_coverage_h_noncvx)
    algo = SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = nothing)

    SCPLib.tune_initial_penalty_weight!(algo, prob, x_ref, u_ref)

    @test algo.w > 0.0
end


function test_scvxstar_tune_with_custom_propagation()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer;
        fun_get_trajectory = scvxstar_coverage_get_trajectory)
    algo = SCPLib.SCvxStar(1, 3; w0 = nothing)

    SCPLib.tune_initial_penalty_weight!(algo, prob, x_ref, u_ref)

    @test algo.w > 0.0
end


function test_scvxstar_tune_invalid_shooting_method()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer)
    algo = SCPLib.SCvxStar(1, 3; w0 = nothing)
    prob.shooting_method = :bogus

    @test_throws UndefVarError SCPLib.tune_initial_penalty_weight!(algo, prob, x_ref, u_ref)
end


function test_scvxstar_invalid_shooting_method_solve()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer;
        set_linearized_constraints! = scvxstar_coverage_set_linearized_constraints!)
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0)
    prob.shooting_method = :bogus

    @test_throws UndefVarError SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 1, verbosity = 0)
end


function test_scvxstar_verbose_run_to_maxiter()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer;
        ng = 1, g_noncvx = scvxstar_coverage_g_noncvx,
        nh = 1, h_noncvx = scvxstar_coverage_h_noncvx)
    # `beta = 1` freezes the penalty weight so 20 iterations stay well conditioned
    algo = SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0,
        beta = 1.0, l1_penalty = true)

    n_callbacks = Ref(0)
    callback = (a, s, it, J0, χ) -> (n_callbacks[] += 1; nothing)

    # `tol_opt = -1` makes the convergence test unreachable, so the run always
    # walks to `maxiter`; 20 iterations also reprints the header.
    solution = redirect_stdout(devnull) do
        SCPLib.solve!(algo, prob, x_ref, u_ref;
            maxiter = 20, tol_opt = -1.0, tol_feas = -1.0,
            verbosity = 2, callback = callback)
    end

    @test solution.status == :MaxIterReached
    @test solution.n_iter == 20
    @test n_callbacks[] == 20
end


function test_scvxstar_feasible_at_maxiter()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer)
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0, beta = 1.0)

    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 2, tol_opt = -1.0, tol_feas = 1e6, verbosity = 0)

    @test solution.status == :Feasible
end


function test_scvxstar_trustregion_control_rejected_steps()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer)
    # thresholds no step can meet, so every iteration is rejected and only the
    # trust-region constraints are refreshed
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0, Δ0_u = 5.0,
        nu = 1, use_trustregion_control = true, rhos = (1e16, 1e16, 1e16))

    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 2, tol_opt = -1.0, tol_feas = -1.0, verbosity = 0)

    @test solution.status == :MaxIterReached
    @test solution.info[:accept] == [false, false]
end


function test_scvxstar_warmstart()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(SCS.Optimizer)
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0, beta = 1.0)

    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 3, tol_opt = -1.0, tol_feas = -1.0, verbosity = 0,
        warmstart_primal = true, warmstart_dual = true)

    @test solution.status == :MaxIterReached
    @test all(isfinite, solution.u)
end


function test_scvxstar_infeasible_subproblem()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer; infeasible = true)
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0)

    solution = redirect_stdout(devnull) do
        SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 3, verbosity = 1)
    end

    @test solution.status == :CPFailed
    @test solution.n_iter == 0
end


function test_scvxstar_callback_on_convergence()
    prob, x_ref, u_ref = make_scvxstar_coverage_problem(Clarabel.Optimizer)
    algo = SCPLib.SCvxStar(1, 3; w0 = 10.0, Δ0 = 5.0)

    n_callbacks = Ref(0)
    callback = (a, s, it, J0, χ) -> (n_callbacks[] += 1; nothing)

    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 50, verbosity = 0, callback = callback)

    @test solution.status == :Optimal
    @test n_callbacks[] >= 1
end


test_scvxstar_invalid_shooting_method_constructor()
test_scvxstar_trustregion_control_without_nu()
test_scvxstar_tune_initial_penalty_weight()
test_scvxstar_tune_with_custom_propagation()
test_scvxstar_tune_invalid_shooting_method()
test_scvxstar_invalid_shooting_method_solve()
test_scvxstar_verbose_run_to_maxiter()
test_scvxstar_feasible_at_maxiter()
test_scvxstar_trustregion_control_rejected_steps()
test_scvxstar_warmstart()
test_scvxstar_infeasible_subproblem()
test_scvxstar_callback_on_convergence()
