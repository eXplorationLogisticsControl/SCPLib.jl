"""Forward-backward shooting regression tests"""

using Clarabel
using JuMP
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


function test_scvxstar_forwardbackward_regressions()
    nx = 1
    nu = 1
    N = 5
    times = LinRange(0.0, 1.0, N)
    x_ref = [0.0 1.0]
    u_ref = zeros(nu, N-1)

    function eom!(dx, x, pu, t)
        dx[1] = pu.u[1]
        return
    end

    objective(x, u) = sum(u)

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        objective,
        times,
        x_ref,
        u_ref;
        shooting_method = :forwardbackward,
    )
    set_silent(prob.model)

    algo = SCPLib.SCvxStar(nx, N; shooting_method = :forwardbackward)
    SCPLib.tune_initial_penalty_weight!(algo, prob, x_ref, u_ref)
    @test !isnothing(algo.w)
    @test isfinite(algo.w)

    g_noncvx(cache, x, u) = [x[1,1] + 2*x[1,end] + sum(u)]
    h_noncvx(cache, x, u) = [x[1,1] - x[1,end] + sum(u)]

    prob_noncvx = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        objective,
        times,
        x_ref,
        u_ref;
        ng = 1,
        g_noncvx = g_noncvx,
        nh = 1,
        h_noncvx = h_noncvx,
        shooting_method = :forwardbackward,
    )
    set_silent(prob_noncvx.model)

    SCPLib.set_linearized_constraints!(prob_noncvx, x_ref, u_ref)
    @test prob_noncvx.lincache.∇g ≈ [1.0 2.0 1.0 1.0 1.0 1.0]
    @test prob_noncvx.lincache.∇h ≈ [1.0 -1.0 1.0 1.0 1.0 1.0]

    z = SCPLib.stack_flatten_variables(prob_noncvx, x_ref, u_ref)
    x_unpacked, u_unpacked = SCPLib.unpack_flattened_variables(prob_noncvx, z)
    @test x_unpacked == x_ref
    @test u_unpacked == u_ref
end


test_scvxstar_forwardbackward_regressions()
