"""Regression tests for forward-backward shooting with non-convex constraints."""

using Clarabel
using JuMP
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

mutable struct ForwardBackwardNonconvxParams
    u::Vector
end

function make_forwardbackward_nonconvx_problem()
    function eom!(dx, x, p, t)
        dx[1] = p.u[1]
        return
    end

    times = [0.0, 1.0, 2.0, 3.0]
    x_ref = zeros(1, 2)
    u_ref = zeros(1, 3)
    objective(x, u) = sum(u .^ 2)
    g_noncvx(cache, x, u) = [sum(x) + sum(u)]

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        ForwardBackwardNonconvxParams([0.0]),
        objective,
        times,
        x_ref,
        u_ref;
        ng = 1,
        g_noncvx = g_noncvx,
        shooting_method = :forwardbackward,
    )
    set_silent(prob.model)

    return prob, x_ref, u_ref
end

@testset "Forward-backward non-convex constraints" begin
    prob, x_ref, u_ref = make_forwardbackward_nonconvx_problem()

    Δz = SCPLib.stack_flatten_variables(prob, x_ref, u_ref)
    x_unpacked, u_unpacked = SCPLib.unpack_flattened_variables(prob, Δz)

    @test length(Δz) == prob.nx * 2 + prob.nu * (prob.N - 1)
    @test size(x_unpacked) == size(x_ref)
    @test size(u_unpacked) == size(u_ref)

    g_dyn_ref, g_ref, h_ref = SCPLib.set_linearized_constraints!(prob, x_ref, u_ref)

    @test size(g_dyn_ref) == (prob.nx, 1)
    @test g_ref == [0.0]
    @test isnothing(h_ref)
    @test prob.lincache.∇g ≈ ones(1, length(Δz))
end

@testset "Forward-backward default penalty tuning" begin
    prob, x_ref, u_ref = make_forwardbackward_nonconvx_problem()
    algo = SCPLib.SCvxStar(prob.nx, prob.N; ng = prob.ng, shooting_method = :forwardbackward)

    SCPLib.tune_initial_penalty_weight!(algo, prob, x_ref, u_ref)

    @test !isnothing(algo.w)
    @test isfinite(algo.w)
end
