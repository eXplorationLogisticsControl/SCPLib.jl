"""Linear dynamics with non-convex path and boundary constraints (`set_linear_dynamics!`)"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


# planar double integrator r̈ = u, state [r; v]
const LINDYN_NX = 4
const LINDYN_NU = 2
const LINDYN_N = 20
const LINDYN_TF = 5.0
const LINDYN_X0 = [-2.0, 0.0, 0.0, 0.0]
const LINDYN_C_OBS = [-1.4, 1.1]     # obstacle center (active at the solution)
const LINDYN_R_OBS = 0.3             # obstacle radius
const LINDYN_RHO = 2.1               # final position on the circle ‖r_N‖ = ρ


"""Exact zero-order-hold discretization of the double integrator"""
function lindyn_analytic_matrices(dt, Nseg)
    I2 = Matrix(1.0I, 2, 2)
    A = [I2 dt*I2; zeros(2, 2) I2]
    B = [dt^2/2*I2; dt*I2]
    return cat(fill(A, Nseg)...; dims = 3), cat(fill(B, Nseg)...; dims = 3), zeros(LINDYN_NX, Nseg)
end


function lindyn_initial_guess()
    N = LINDYN_N
    xf = [LINDYN_RHO, 0.5, 0.0, 0.0]
    x_ref = reduce(hcat, [LINDYN_X0 + (k-1)/(N-1) * (xf - LINDYN_X0) for k in 1:N])
    u_ref = zeros(LINDYN_NU, N-1)
    return x_ref, u_ref
end


"""
Double integrator with an obstacle at every node (`nh = N`) and a final position on a circle
(`ng = 1`). `eom!` counts its calls.
"""
function make_lindyn_problem(; frozen::Bool = false, analytic::Bool = false, kwargs...)
    nx, nu, N = LINDYN_NX, LINDYN_NU, LINDYN_N
    times = LinRange(0.0, LINDYN_TF, N)
    x_ref, u_ref = lindyn_initial_guess()

    eom_calls = Ref(0)
    function eom!(dx, x, p, t)
        eom_calls[] += 1
        dx[1:2] .= x[3:4]
        dx[3:4] .= p.u
        return
    end

    h_noncvx(cache, x, u) = [LINDYN_R_OBS - norm(x[1:2, k] - LINDYN_C_OBS) for k in 1:N]
    g_noncvx(cache, x, u) = [sum(abs2, x[1:2, N]) - LINDYN_RHO^2]

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer, eom!, nothing, (x, u) -> sum(abs2, u), times, x_ref, u_ref;
        ng = 1, g_noncvx = g_noncvx, nh = N, h_noncvx = h_noncvx, kwargs...,
    )
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][:, 1] .== LINDYN_X0)
    @constraint(prob.model, prob.model[:x][3:4, N] .== 0.0)
    @constraint(prob.model, [k = 1:N-1], [2.0; prob.model[:u][:, k]] in SecondOrderCone())

    if analytic
        Φ_A, Φ_B, Φ_c = lindyn_analytic_matrices(times[2] - times[1], N-1)
        SCPLib.set_linear_dynamics!(prob; Φ_A, Φ_B, Φ_c)
    elseif frozen
        SCPLib.set_linear_dynamics!(prob)
    end
    return prob, x_ref, u_ref, eom_calls
end


function test_linear_dynamics_discretization()
    rng = Xoshiro(0)
    N = LINDYN_N
    dt = LINDYN_TF / (N-1)
    Φ_A, Φ_B, Φ_c = lindyn_analytic_matrices(dt, N-1)

    # discretized once about zeros: exact ZOH matrices
    prob, _, _, _ = make_lindyn_problem(; frozen = true)
    @test prob.lincache.Φ_A ≈ Φ_A atol = 1e-10
    @test prob.lincache.Φ_B ≈ Φ_B atol = 1e-10
    @test prob.lincache.Φ_c ≈ Φ_c atol = 1e-10

    # the hook reloads the frozen matrices, whatever the reference
    x, u = randn(rng, LINDYN_NX, N), randn(rng, LINDYN_NU, N-1)
    prob.lincache.Φ_A .= 0.0
    g_hook = prob.set_dynamics_cache!(prob, x, u)
    @test prob.lincache.Φ_A ≈ Φ_A atol = 1e-10

    # defects agree with ODE propagation; matrices agree with the default linearization at a random reference
    prob_ode, _, _, _ = make_lindyn_problem()
    g_ode = SCPLib.set_dynamics_cache!(prob_ode, x, u)
    @test prob_ode.lincache.Φ_A ≈ Φ_A atol = 1e-10
    @test prob_ode.lincache.Φ_B ≈ Φ_B atol = 1e-10
    @test prob_ode.lincache.Φ_c ≈ Φ_c atol = 1e-10
    @test g_hook ≈ g_ode atol = 1e-10
    @test SCPLib.get_dynamics_defects(prob, x, u) ≈ SCPLib.get_trajectory(prob_ode, x, u)[2] atol = 1e-10

    # u_bias is folded into Φ_c
    u_bias = randn(rng, LINDYN_NU, N)
    prob_b, _, _, _ = make_lindyn_problem(; frozen = true, u_bias = u_bias)
    prob_b_ode, _, _, _ = make_lindyn_problem(; u_bias = u_bias)
    @test prob_b.lincache.Φ_c ≈ reduce(hcat, [Φ_B[:, :, k] * u_bias[:, k] for k in 1:N-1]) atol = 1e-10
    @test SCPLib.get_dynamics_defects(prob_b, x, u) ≈ SCPLib.get_trajectory(prob_b_ode, x, u)[2] atol = 1e-10
end


function test_linear_dynamics_impulsive()
    rng = Xoshiro(1)
    nx, nu, N = LINDYN_NX, LINDYN_NU, LINDYN_N
    times = LinRange(0.0, LINDYN_TF, N)
    dt = times[2] - times[1]
    eom!(dx, x, p, t) = (dx[1:2] .= x[3:4]; dx[3:4] .= 0.0; nothing)
    dfdu(x, u, t) = [zeros(2, 2); Matrix(1.0I, 2, 2)]
    make() = SCPLib.ImpulsiveProblem(
        Clarabel.Optimizer, eom!, nothing, (x, u) -> sum(abs2, u), times, zeros(nx, N), zeros(nu, N);
        dfdu = dfdu,
    )
    prob = SCPLib.set_linear_dynamics!(make())
    prob_ode = make()

    Φ_A, _, _ = lindyn_analytic_matrices(dt, N-1)
    @test prob.lincache.Φ_A ≈ Φ_A atol = 1e-10
    @test prob.lincache.Φ_B ≈ cat([Φ_A[:, :, k] * dfdu(nothing, nothing, nothing) for k in 1:N-1]...; dims = 3) atol = 1e-10
    x, u = randn(rng, nx, N), randn(rng, nu, N)     # Nu = N: u[:, N] does not enter the dynamics
    @test SCPLib.get_dynamics_defects(prob, x, u) ≈ SCPLib.get_trajectory(prob_ode, x, u)[2] atol = 1e-10
end


function test_linear_dynamics_impulsive_solve()
    rng = Xoshiro(2)
    nx, nu, N = LINDYN_NX, LINDYN_NU, LINDYN_N
    times = LinRange(0.0, LINDYN_TF, N)
    eom_calls = Ref(0)
    eom!(dx, x, p, t) = (eom_calls[] += 1; dx[1:2] .= x[3:4]; dx[3:4] .= 0.0; nothing)
    dfdu(x, u, t) = [zeros(2, 2); Matrix(1.0I, 2, 2)]
    u_bias = 0.1 * randn(rng, nu, N)
    x_ref, _ = lindyn_initial_guess()
    u_ref = zeros(nu, N)
    function make()
        prob = SCPLib.ImpulsiveProblem(
            Clarabel.Optimizer, eom!, nothing, (x, u) -> sum(abs2, u), times, x_ref, u_ref;
            dfdu = dfdu, u_bias = u_bias,
            nh = N, h_noncvx = (cache, x, u) -> [LINDYN_R_OBS - norm(x[1:2, k] - LINDYN_C_OBS) for k in 1:N],
        )
        set_silent(prob.model)
        @constraint(prob.model, prob.model[:x][:, 1] .== LINDYN_X0)
        @constraint(prob.model, prob.model[:x][:, N] .== [LINDYN_RHO, 0.5, 0.0, 0.0])
        return prob
    end
    run(prob) = SCPLib.solve!(SCPLib.SCvxStar(nx, N; nh = N, w0 = 1e2), prob, copy(x_ref), copy(u_ref);
                              maxiter = 10, verbosity = 0)

    sol_ode = run(make())
    prob = SCPLib.set_linear_dynamics!(make())
    calls_before = eom_calls[]
    sol = run(prob)
    @test eom_calls[] == calls_before       # no ODE propagation during solve!
    @test sol.info[:J0] ≈ sol_ode.info[:J0] rtol = 1e-6
    @test sol.info[:χ] ≈ sol_ode.info[:χ] rtol = 1e-6 atol = 1e-9
    @test SCPLib.get_dynamics_defects(prob, sol.x, sol.u) ≈ SCPLib.get_trajectory(make(), sol.x, sol.u)[2] atol = 1e-9
end


function test_linear_dynamics_scvxstar()
    nx, N = LINDYN_NX, LINDYN_N
    make_algo() = SCPLib.SCvxStar(nx, N; ng = 1, nh = N, w0 = 1e2)
    run(prob, x_ref, u_ref) = SCPLib.solve!(make_algo(), prob, copy(x_ref), copy(u_ref); maxiter = 15, verbosity = 0)

    prob_ode, x_ref, u_ref, _ = make_lindyn_problem()
    sol_ode = run(prob_ode, x_ref, u_ref)

    for (label, kw) in (("discretized", (; frozen = true)), ("analytic", (; analytic = true)))
        @testset "$label" begin
            prob, x_ref, u_ref, eom_calls = make_lindyn_problem(; kw...)
            calls_before = eom_calls[]
            sol = run(prob, x_ref, u_ref)
            @test eom_calls[] == calls_before       # no ODE propagation during solve!
            @test sol.status != :CPFailed
            # same iterates as with ODE propagation
            @test length(sol.info[:J0]) == length(sol_ode.info[:J0])
            @test sol.info[:J0] ≈ sol_ode.info[:J0] rtol = 1e-6
            @test sol.info[:χ] ≈ sol_ode.info[:χ] rtol = 1e-6 atol = 1e-9
            @test sol.x ≈ sol_ode.x atol = 1e-6
            # the frozen defects are the true ODE defects
            @test SCPLib.get_dynamics_defects(prob, sol.x, sol.u) ≈
                  SCPLib.get_trajectory(prob_ode, sol.x, sol.u)[2] atol = 1e-9
        end
    end
end


function test_linear_dynamics_other_algorithms()
    nx, N = LINDYN_NX, LINDYN_N
    algos = [
        "SCvx"        => () -> SCPLib.SCvx(nx, N; w = 1e3),
        "FixedTRWSCP" => () -> SCPLib.FixedTRWSCP(nx, N, 0.5, 1e3),
        "ProxLinear"  => () -> SCPLib.ProxLinear(1e3, 1e0),
    ]
    for (name, make_algo) in algos
        @testset "$name" begin
            prob, x_ref, u_ref, eom_calls = make_lindyn_problem(; frozen = true)
            calls_before = eom_calls[]
            sol = SCPLib.solve!(make_algo(), prob, copy(x_ref), copy(u_ref); maxiter = 5, verbosity = 0)
            @test eom_calls[] == calls_before
            @test sol.status != :CPFailed
        end
    end
end


function test_linear_dynamics_converges()
    # the obstacle (path constraint) and the final circle (boundary constraint) are both active
    nx, N = LINDYN_NX, LINDYN_N
    prob, x_ref, u_ref, eom_calls = make_lindyn_problem(; frozen = true)
    calls_before = eom_calls[]
    sol = SCPLib.solve!(SCPLib.SCvxStar(nx, N; ng = 1, nh = N, w0 = 1e2), prob, copy(x_ref), copy(u_ref);
                        maxiter = 100, verbosity = 0)
    @test eom_calls[] == calls_before
    @test sol.status in (:Optimal, :Feasible)
    @test sol.info[:χ][end] <= 1e-6
    @test maximum(abs, SCPLib.get_dynamics_defects(prob, sol.x, sol.u)) <= 1e-6
    h = prob.h_noncvx(prob.lincache, sol.x, sol.u)
    @test maximum(h) <= 1e-6                # obstacle avoided ...
    @test maximum(h) >= -1e-3               # ... and touched
    @test abs(prob.g_noncvx(prob.lincache, sol.x, sol.u)[1]) <= 1e-6
end


function test_linear_dynamics_repeated_solve()
    nx, N = LINDYN_NX, LINDYN_N
    prob, x_ref, u_ref, eom_calls = make_lindyn_problem(; frozen = true)
    calls_before = eom_calls[]
    sols = [SCPLib.solve!(SCPLib.SCvxStar(nx, N; ng = 1, nh = N, w0 = 1e2), prob, copy(x_ref), copy(u_ref);
                          maxiter = 5, verbosity = 0) for _ in 1:2]
    @test sols[1].info[:J0] ≈ sols[2].info[:J0] rtol = 1e-8
    @test eom_calls[] == calls_before
end


function test_linear_dynamics_errors()
    N = LINDYN_N
    Φ_A, Φ_B, Φ_c = lindyn_analytic_matrices(1.0, N-1)
    prob, _, _, _ = make_lindyn_problem()
    @test_throws AssertionError SCPLib.set_linear_dynamics!(prob; Φ_A, Φ_B)
    @test_throws AssertionError SCPLib.set_linear_dynamics!(prob; Φ_A = Φ_A[:, :, 1:end-1], Φ_B, Φ_c)
    @test_throws AssertionError SCPLib.set_linear_dynamics!(prob; Φ_A, Φ_B = Φ_B[:, 1:1, :], Φ_c)
    @test isnothing(prob.fun_get_trajectory)        # failed calls leave the problem unchanged

    @test_throws AssertionError SCPLib.set_linear_dynamics!(prob; u_ref = zeros(LINDYN_NU, N))
    @test isnothing(prob.fun_get_trajectory)        # failed calls leave the problem unchanged

    # overwriting user hooks warns; re-freezing a frozen problem does not
    prob_hook, _, _, _ = make_lindyn_problem(; fun_get_trajectory = (p, x, u) -> SCPLib.get_trajectory(p, x, u))
    @test_logs (:warn, r"overwrites") SCPLib.set_linear_dynamics!(prob_hook; Φ_A, Φ_B, Φ_c)
    @test_logs SCPLib.set_linear_dynamics!(prob_hook; Φ_A, Φ_B, Φ_c)

    # user-supplied matrices with a control bias warn
    prob_bias, _, _, _ = make_lindyn_problem(; u_bias = ones(LINDYN_NU, N))
    @test_logs (:warn, r"u_bias") SCPLib.set_linear_dynamics!(prob_bias; Φ_A, Φ_B, Φ_c)

    # forward-backward shooting is not supported
    times = LinRange(0.0, 1.0, 3)
    prob_fb = SCPLib.ContinuousProblem(
        Clarabel.Optimizer, (dx, x, p, t) -> (dx .= p.u; nothing), nothing, (x, u) -> sum(u),
        times, zeros(1, 2), zeros(1, 2); shooting_method = :forwardbackward,
    )
    @test_throws AssertionError SCPLib.set_linear_dynamics!(prob_fb)
end


test_linear_dynamics_discretization()
test_linear_dynamics_impulsive()
test_linear_dynamics_impulsive_solve()
test_linear_dynamics_scvxstar()
test_linear_dynamics_other_algorithms()
test_linear_dynamics_converges()
test_linear_dynamics_repeated_solve()
test_linear_dynamics_errors()
