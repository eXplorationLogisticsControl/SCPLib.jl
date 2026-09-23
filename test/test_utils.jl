"""Unit tests for utility functions"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


function test_ensemble_sim_id()
    # OrdinaryDiffEq v6 passes the integer index
    @test SCPLib.ensemble_sim_id(3) == 3
    # OrdinaryDiffEq v7 passes a context object
    @test SCPLib.ensemble_sim_id((; sim_id = 7)) == 7
end


function test_ensemble_trajectories()
    # a plain vector of solutions passes through
    sols_vector = [1.0, 2.0, 3.0]
    @test SCPLib.ensemble_trajectories(sols_vector) === sols_vector

    # an ensemble solution unwraps to `sols.u`
    ode_problem = ODEProblem((dx, x, p, t) -> (dx[1] = -x[1]; return), [1.0], (0.0, 1.0))
    sols = solve(
        EnsembleProblem(ode_problem),
        Tsit5(),
        SciMLBase.EnsembleSerial();
        trajectories = 2,
    )
    @test SCPLib.ensemble_trajectories(sols) === sols.u
    @test length(SCPLib.ensemble_trajectories(sols)) == 2
end


function test_message_accept_step()
    @test SCPLib.message_accept_step(true) == "yes"
    @test SCPLib.message_accept_step(false) == "no "
end


function test_delete_noncvx_referencs(;verbosity::Int = 0)
    # -------------------- minimal problem -------------------- #
    nx = 2
    nu = 1
    N = 4

    function eom!(dx, x, pu, t)
        (; params, u) = pu
        dx[1] = x[2]
        dx[2] = u[1]
        return
    end

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        (x, u) -> sum(u.^2),
        LinRange(0.0, 1.0, N),
        zeros(nx, N),
        zeros(nu, N-1),
    )
    if verbosity == 0
        set_silent(prob.model)
    end

    n_constraints_0 = num_constraints(prob.model; count_variable_in_set_constraints = true)

    # a single reference and a container of references
    @constraint(prob.model, constraint_scalar, sum(prob.model[:u]) <= 1.0)
    @constraint(prob.model, constraint_container[k in 1:N], prob.model[:x][1,k] <= 1.0)
    @test num_constraints(prob.model; count_variable_in_set_constraints = true) ==
        n_constraints_0 + 1 + N

    SCPLib.delete_noncvx_referencs!(prob, [:constraint_scalar, :constraint_container])
    @test num_constraints(prob.model; count_variable_in_set_constraints = true) == n_constraints_0
    @test !haskey(object_dictionary(prob.model), :constraint_scalar)
    @test !haskey(object_dictionary(prob.model), :constraint_container)
end


function test_get_solutions(;verbosity::Int = 0)
    # min x1 + 2*x2  s.t.  x1 + x2 == 1, 0.25 <= x1 <= 0.75  =>  x1 = 0.75, x2 = 0.25
    model = Model(Clarabel.Optimizer)
    if verbosity == 0
        set_silent(model)
    end
    @variable(model, x[1:2])
    @constraint(model, constraint_sum, x[1] + x[2] == 1.0)
    @constraint(model, constraint_lb, x[1] >= 0.25)
    @constraint(model, constraint_ub, x[1] <= 0.75)
    @objective(model, Min, x[1] + 2 * x[2])
    optimize!(model)
    @test termination_status(model) == OPTIMAL

    variable_primal = SCPLib.get_primal_variables(model)
    @test length(variable_primal) == 2
    @test variable_primal[x[1]] ≈ 0.75 atol=1e-6
    @test variable_primal[x[2]] ≈ 0.25 atol=1e-6

    constraint_solution = SCPLib.get_constraint_solutions(model)
    @test length(constraint_solution) == 3
    @test constraint_solution[constraint_sum][1] ≈ 1.0 atol=1e-6
    @test constraint_solution[constraint_ub] == (value(constraint_ub), dual(constraint_ub))
end


function test_set_optimal_start_values()
    # Clarabel rejects the start-value attributes, so warm-start an optimizer-free model
    model = Model()
    @variable(model, x[1:2])
    @constraint(model, constraint_sum, x[1] + x[2] == 1.0)
    @constraint(model, constraint_ub, 2 * x[1] <= 1.5)

    variable_primal = Dict(x[1] => 0.75, x[2] => 0.25)
    constraint_solution = Dict(constraint_sum => (1.0, -1.0), constraint_ub => (1.5, -0.5))

    SCPLib.set_optimal_start_values(variable_primal, constraint_solution)
    @test start_value(x[1]) ≈ 0.75
    @test start_value(x[2]) ≈ 0.25
    @test start_value(constraint_sum) ≈ 1.0
    @test dual_start_value(constraint_sum) ≈ -1.0
    @test start_value(constraint_ub) ≈ 1.5
    @test dual_start_value(constraint_ub) ≈ -0.5
end


function test_get_constraint_solutions_unsolved(;verbosity::Int = 0)
    # querying an unsolved model throws internally and is skipped with a log message
    model = Model(Clarabel.Optimizer)
    if verbosity == 0
        set_silent(model)
    end
    @variable(model, x[1:2])
    @constraint(model, constraint_sum, x[1] + x[2] == 1.0)

    constraint_solution = SCPLib.get_constraint_solutions(model)
    @test isempty(constraint_solution)
end


test_ensemble_sim_id()
test_ensemble_trajectories()
test_message_accept_step()
test_delete_noncvx_referencs(verbosity = verbosity)
test_get_solutions(verbosity = verbosity)
test_set_optimal_start_values()
test_get_constraint_solutions_unsolved(verbosity = verbosity)
