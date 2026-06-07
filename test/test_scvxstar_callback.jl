"""Test callback invocation compatibility."""

using Clarabel
using JuMP

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

mutable struct CallbackTestParams
    u::Vector
end

function make_callback_test_problem()
    function eom!(dx, x, p, t)
        dx[1] = p.u[1]
        return
    end

    times = [0.0, 1.0, 2.0]
    x_ref = zeros(1, 3)
    u_ref = zeros(1, 2)
    objective(x, u) = 0.0

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        CallbackTestParams([0.0]),
        objective,
        times,
        x_ref,
        u_ref,
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][:,1] .== 0.0)
    @constraint(prob.model, prob.model[:x][:,end] .== 0.0)
    @constraint(prob.model, prob.model[:u] .== 0.0)

    return prob, x_ref, u_ref
end

@testset "callback compatibility" begin
    prob, x_ref, u_ref = make_callback_test_problem()
    legacy_calls = Ref(0)
    legacy_solution_type = Ref{DataType}()
    legacy_callback(solution) = begin
        legacy_calls[] += 1
        legacy_solution_type[] = typeof(solution)
        return nothing
    end

    algo = SCPLib.SCvxStar(1, 3; w0 = 1.0)
    SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 1, verbosity = 0, callback = legacy_callback)

    @test legacy_calls[] == 1
    @test legacy_solution_type[] == SCPLib.SCvxStarSolution

    prob, x_ref, u_ref = make_callback_test_problem()
    callback_args = Ref{Tuple}()
    new_callback(algo, solution, iteration, J0, chi) = begin
        callback_args[] = (typeof(algo), typeof(solution), iteration, J0, chi)
        return nothing
    end

    algo = SCPLib.SCvxStar(1, 3; w0 = 1.0)
    SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 1, verbosity = 0, callback = new_callback)

    @test callback_args[][1] == SCPLib.SCvxStar
    @test callback_args[][2] == SCPLib.SCvxStarSolution
    @test callback_args[][3] == 1
    @test callback_args[][4] isa Real
    @test callback_args[][5] isa Real
end