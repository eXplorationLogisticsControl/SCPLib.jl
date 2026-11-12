"""Calling `solve!` more than once on the same problem (TODO m2) and dual warm-starting (TODO M6)"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using SCS
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


"""
Single-state integrator `ẋ = 1 + u` steered from 0 to 1, with one non-convex equality and one
non-convex inequality constraint on the middle node. The returned reference `u_ref = 1` violates
the dynamics, so the first step is accepted and the next iteration re-linearizes.
"""
function make_repeated_solve_problem(optimizer = Clarabel.Optimizer)
    nx, nu, N = 1, 1, 3
    x_ref = [0.0 0.5 1.0]
    u_ref = ones(nu, N-1)

    function eom!(dx, x, pu, t)
        dx[1] = 1.0 + pu.u[1]
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
        ng = 1,
        g_noncvx = (cache, x, u) -> [x[1,2]^2 - 0.25],
        nh = 1,
        h_noncvx = (cache, x, u) -> [x[1,2]^2 - 1.0],
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,end] == 1.0)
    return prob, x_ref, u_ref
end


"""Number of variables and constraints in the problem's JuMP model"""
model_size(prob) = (num_variables(prob.model),
                    num_constraints(prob.model; count_variable_in_set_constraints = true))


const REPEATED_SOLVE_ALGORITHMS = [
    ("SCvx*", () -> SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0)),
    ("SCvx* (l1 penalty, control trust region)",
        () -> SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0,
                              l1_penalty = true, use_trustregion_control = true)),
    ("SCvx", () -> SCPLib.SCvx(1, 3; w = 1e3, Δ0 = 5.0)),
    ("SCvx (control trust region)", () -> SCPLib.SCvx(1, 3; w = 1e3, Δ0 = 5.0, use_trustregion_control = true)),
    ("ProxLinear", () -> SCPLib.ProxLinear(1e2, 1e0)),
    ("FixedTRW", () -> SCPLib.FixedTRWSCP(1, 3, 5.0, 1e3)),
]


function solve_repeated(algo, prob, x_ref, u_ref; kwargs...)
    return SCPLib.solve!(algo, prob, copy(x_ref), copy(u_ref); maxiter = 5, verbosity = 0, kwargs...)
end


function test_solve_twice()
    for (name, make_algo) in REPEATED_SOLVE_ALGORITHMS
        @testset "$name" begin
            prob, x_ref, u_ref = make_repeated_solve_problem()
            nl_references = copy(prob.model_nl_references)
            size_0 = model_size(prob)

            solution_1 = solve_repeated(make_algo(), prob, x_ref, u_ref)
            size_1 = model_size(prob)
            solution_2 = solve_repeated(make_algo(), prob, x_ref, u_ref)

            # same algorithm settings give the same result, and nothing accumulates in the model
            @test solution_2.status == solution_1.status
            @test solution_2.x ≈ solution_1.x atol = 1e-6
            @test solution_2.u ≈ solution_1.u atol = 1e-6
            @test model_size(prob) == size_1
            @test prob.model_nl_references == nl_references

            # resetting leaves only what the user defined
            SCPLib.reset_solver_model!(prob)
            @test model_size(prob) == size_0
            @test isempty(get(prob.model.ext, SCPLib.SOLVER_OBJECTS_KEY, Any[]))
        end
    end
end


function test_switch_algorithms()
    prob, x_ref, u_ref = make_repeated_solve_problem()
    nl_references = copy(prob.model_nl_references)
    size_0 = model_size(prob)

    # each algorithm on a fresh problem, as reference
    reference = Dict(name => solve_repeated(make_algo(), make_repeated_solve_problem()...)
                     for (name, make_algo) in REPEATED_SOLVE_ALGORITHMS)

    # every algorithm in turn on the same problem, then again in reverse order
    for (name, make_algo) in vcat(REPEATED_SOLVE_ALGORITHMS, reverse(REPEATED_SOLVE_ALGORITHMS))
        solution = solve_repeated(make_algo(), prob, x_ref, u_ref)
        @test solution.status == reference[name].status
        @test solution.u ≈ reference[name].u atol = 1e-6
    end
    @test prob.model_nl_references == nl_references

    SCPLib.reset_solver_model!(prob)
    @test model_size(prob) == size_0
end


function test_solve_after_error()
    prob, x_ref, u_ref = make_repeated_solve_problem()
    algo = SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0, l1_penalty = true)

    # a callback that throws leaves the linearized constraints of iteration 1 in the model
    failing_callback = (a, s, it, J0, χ) -> error("stop")
    @test_throws ErrorException solve_repeated(algo, prob, x_ref, u_ref; callback = failing_callback)
    @test haskey(object_dictionary(prob.model), :constraint_dynamics)

    solution = solve_repeated(SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0, l1_penalty = true),
                              prob, x_ref, u_ref)
    reference = solve_repeated(SCPLib.SCvxStar(1, 3; ng = 1, nh = 1, w0 = 10.0, Δ0 = 5.0, l1_penalty = true),
                               make_repeated_solve_problem()...)
    @test solution.status == reference.status
    @test solution.u ≈ reference.u atol = 1e-6
end


function test_warmstart_after_accepted_step()
    for (name, make_algo) in REPEATED_SOLVE_ALGORITHMS
        name == "FixedTRW" && continue       # no warm-start option
        @testset "$name" begin
            prob, x_ref, u_ref = make_repeated_solve_problem(SCS.Optimizer)
            solution = solve_repeated(make_algo(), prob, x_ref, u_ref;
                tol_opt = -1.0, warmstart_primal = true, warmstart_dual = true)

            # the first step is accepted, so iteration 2 re-linearizes and then warm-starts
            @test solution.status ∈ (:Feasible, :MaxIterReached)
            @test length(solution.info[:J0]) ≥ 2
            accept = get(solution.info, :accept, Bool[])     # ProxLinear accepts every step
            if !isempty(accept)
                @test first(accept)
            end
            @test all(isfinite, solution.u)
        end
    end
end


"""Stored start values whose variable or constraint was deleted are skipped"""
function test_set_optimal_start_values_skips_deleted()
    model = Model()
    x = @variable(model)
    y = @variable(model)
    c_kept = @constraint(model, x + y <= 1.0)
    c_deleted = @constraint(model, x - y <= 1.0)
    variable_primal = Dict(x => 0.25, y => 0.5)
    constraint_solution = Dict(c_kept => (0.75, -1.0), c_deleted => (-0.25, -2.0))

    delete(model, c_deleted)
    delete(model, y)
    SCPLib.set_optimal_start_values(variable_primal, constraint_solution)

    @test start_value(x) == 0.25
    @test start_value(c_kept) == 0.75
    @test dual_start_value(c_kept) == -1.0
end


@testset "Solve twice" begin
    test_solve_twice()
end
@testset "Switch algorithms" begin
    test_switch_algorithms()
end
@testset "Solve after error" begin
    test_solve_after_error()
end
@testset "Warm start after accepted step" begin
    test_warmstart_after_accepted_step()
end
@testset "Start values of deleted references" begin
    test_set_optimal_start_values_skips_deleted()
end
