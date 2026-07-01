"""Test SCvx* callback state ordering."""

using Clarabel
using JuMP
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

struct CallbackStateParams end

function make_callback_state_problem()
    function eom!(dx, x, pu, t)
        dx[1] = 0.0
        return
    end

    times = [0.0, 1.0]
    x_ref = zeros(1, 2)
    u_ref = zeros(1, 1)
    objective(x, u) = 0.0

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        CallbackStateParams(),
        objective,
        times,
        x_ref,
        u_ref,
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][1, 1] == 0.0)
    @constraint(prob.model, prob.model[:x][1, 2] == 1.0)
    @constraint(prob.model, prob.model[:u][1, 1] == 0.0)

    return prob, x_ref, u_ref
end

@testset "SCvxStar callback runs after state updates" begin
    prob, x_ref, u_ref = make_callback_state_problem()

    initial_w = 10.0
    callback_w = 1000.0
    callback_calls = Ref(0)
    callback_seen_w = Ref(NaN)

    function callback(algo, solution, iteration, J0, chi)
        callback_calls[] += 1
        callback_seen_w[] = algo.w
        algo.w = callback_w
        return nothing
    end

    algo = SCPLib.SCvxStar(1, 2; w0 = initial_w, Δ0 = 2.0, beta = 2.0)
    solution = SCPLib.solve!(
        algo,
        prob,
        x_ref,
        u_ref;
        maxiter = 1,
        verbosity = 0,
        callback = callback,
    )

    _, g_dynamics = SCPLib.get_trajectory(prob, solution.x, solution.u)

    @test callback_calls[] == 1
    @test solution.info[:accept] == [true]
    @test only(g_dynamics) ≈ 1.0 atol = 1e-8
    @test only(algo.λ_dyn) ≈ initial_w * only(g_dynamics) atol = 1e-8
    @test callback_seen_w[] ≈ initial_w * algo.beta
    @test algo.w == callback_w

    @test solution.info[:J0][end] ≈ 0.0 atol=1e-8
end