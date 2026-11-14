"""Every algorithm evaluates the dynamics defects through `prob.fun_get_trajectory` when it is set"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


"""
Single integrator `ẋ = u` steered from 0 to 1. `eom!` counts its calls; `eom_aug!` is
separate, so only the defect evaluation can call `eom!`. `fun_get_trajectory` evaluates the
(exact) defects algebraically and counts its calls.
"""
function make_fun_get_trajectory_problem()
    nx, nu, N = 1, 1, 4
    times = LinRange(0.0, 1.0, N)
    dt = times[2] - times[1]
    x_ref = zeros(nx, N)
    u_ref = zeros(nu, N-1)

    eom_calls = Ref(0)
    hook_calls = Ref(0)

    function eom!(dx, x, pu, t)
        eom_calls[] += 1
        dx[1] = pu.u[1]
        return
    end

    function eom_aug!(dx_aug, x_aug, pu, t)
        dx_aug[1] = pu.u[1]     # state
        dx_aug[2] = 0.0         # Φ_A
        dx_aug[3] = 1.0         # Φ_B
        return
    end

    function fun_get_trajectory(prob, x, u)
        hook_calls[] += 1
        g = reduce(hcat, [x[:, k+1] - (x[:, k] + dt * u[:, k]) for k in 1:prob.N-1])
        return nothing, g
    end

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        (x, u) -> sum(u.^2),
        times,
        x_ref,
        u_ref;
        eom_aug! = eom_aug!,
        fun_get_trajectory = fun_get_trajectory,
    )
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][1, 1] == 0.0)
    @constraint(prob.model, prob.model[:x][1, end] == 1.0)
    return prob, x_ref, u_ref, eom_calls, hook_calls
end


function test_fun_get_trajectory_all_algorithms()
    nx, N = 1, 4
    algos = [
        "SCvxStar"    => () -> SCPLib.SCvxStar(nx, N; w0 = 1e2, Δ0 = 1.0),
        "SCvx"        => () -> SCPLib.SCvx(nx, N; Δ0 = 1.0, w = 1e2),
        "FixedTRWSCP" => () -> SCPLib.FixedTRWSCP(nx, N, 1.0, 1e6),
        "ProxLinear"  => () -> SCPLib.ProxLinear(1e2, 1e0),
    ]
    for (name, make_algo) in algos
        @testset "$name" begin
            prob, x_ref, u_ref, eom_calls, hook_calls = make_fun_get_trajectory_problem()
            solution = SCPLib.solve!(make_algo(), prob, x_ref, u_ref; maxiter = 5, verbosity = 0)
            @test hook_calls[] > 0
            @test eom_calls[] == 0
            @test solution.status != :CPFailed
            # exact optimum of min Σu² with x(0) = 0, x(1) = 1: u ≡ 1
            @test solution.u ≈ ones(1, N-1) atol = 1e-3
        end
    end
end


test_fun_get_trajectory_all_algorithms()
