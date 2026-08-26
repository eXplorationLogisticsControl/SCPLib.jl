"""Forward-backward compatibility tests for SCP algorithms"""

using Clarabel
using JuMP
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


function build_forwardbackward_problem()
    nx = 1
    nu = 1
    N = 4
    times = LinRange(0.0, 1.0, N)
    x_ref = [0.0 0.0]
    u_ref = zeros(nu, N-1)

    function eom!(dx, x, pu, t)
        (; u) = pu
        dx[1] = u[1]
        return
    end

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        (x, u) -> sum(u.^2),
        times,
        copy(x_ref),
        copy(u_ref);
        shooting_method = :forwardbackward,
    )
    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,2] == 0.0)
    set_silent(prob.model)

    return prob, copy(x_ref), copy(u_ref), nx, N
end


function test_forwardbackward_scvx()
    prob, x_ref, u_ref, nx, N = build_forwardbackward_problem()
    solution = SCPLib.solve!(
        SCPLib.SCvx(nx, N; w = 1e3),
        prob,
        x_ref,
        u_ref;
        maxiter = 1,
        verbosity = 0,
    )

    @test size(solution.x) == (nx, 2)
    @test size(solution.u) == size(u_ref)
    @test solution.status != :CPFailed
end


function test_forwardbackward_proxlinear()
    prob, x_ref, u_ref, nx, _ = build_forwardbackward_problem()
    solution = SCPLib.solve!(
        SCPLib.ProxLinear(),
        prob,
        x_ref,
        u_ref;
        maxiter = 1,
        verbosity = 0,
    )

    @test size(solution.x) == (nx, 2)
    @test size(solution.u) == size(u_ref)
    @test solution.status != :CPFailed
end


function test_forwardbackward_fixedtrw()
    prob, x_ref, u_ref, nx, N = build_forwardbackward_problem()
    solution = SCPLib.solve!(
        SCPLib.FixedTRWSCP(nx, N, 0.1, 1e3),
        prob,
        x_ref,
        u_ref;
        maxiter = 1,
        verbosity = 0,
    )

    @test size(solution.x) == (nx, 2)
    @test size(solution.u) == size(u_ref)
    @test solution.status != :CPFailed
end


test_forwardbackward_scvx()
test_forwardbackward_proxlinear()
test_forwardbackward_fixedtrw()
