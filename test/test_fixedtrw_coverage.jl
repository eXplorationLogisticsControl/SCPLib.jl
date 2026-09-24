"""Exercise the fixed trust-region & weight branches that the physical test problem never reaches"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


fixedtrw_coverage_g_noncvx(cache, x, u) = [x[1,2]^2 - 0.25]
fixedtrw_coverage_h_noncvx(cache, x, u) = [x[1,2]^2 - 1.0]


"""Single-state integrator `ẋ = u` steered from 0 to 1"""
function make_fixedtrw_coverage_problem(optimizer;
    ng::Int = 0, g_noncvx = nothing,
    nh::Int = 0, h_noncvx = nothing,
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
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,end] == 1.0)
    if infeasible
        @constraint(prob.model, prob.model[:x][1,1] == 1.0)
    end
    return prob, x_ref, u_ref
end


function test_fixedtrw_verbose_run_to_maxiter()
    prob, x_ref, u_ref = make_fixedtrw_coverage_problem(Clarabel.Optimizer;
        ng = 1, g_noncvx = fixedtrw_coverage_g_noncvx,
        nh = 1, h_noncvx = fixedtrw_coverage_h_noncvx)
    # the weight is given explicitly; leaving it at `nothing` derives it from
    # `tol_feas`, which the negative tolerance below would turn negative
    algo = SCPLib.FixedTRWSCP(1, 3, 5.0, 1e6)

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


function test_fixedtrw_feasible_at_maxiter()
    prob, x_ref, u_ref = make_fixedtrw_coverage_problem(Clarabel.Optimizer)
    algo = SCPLib.FixedTRWSCP(1, 3, 5.0, 1e6)

    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        maxiter = 2, tol_opt = -1.0, tol_feas = 1e6, verbosity = 0)

    @test solution.status == :Feasible
end


function test_fixedtrw_infeasible_subproblem()
    prob, x_ref, u_ref = make_fixedtrw_coverage_problem(Clarabel.Optimizer; infeasible = true)
    algo = SCPLib.FixedTRWSCP(1, 3, 5.0, 1e6)

    solution = redirect_stdout(devnull) do
        SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 3, verbosity = 1)
    end

    @test solution.status == :CPFailed
    @test solution.n_iter == 0
end


test_fixedtrw_verbose_run_to_maxiter()
test_fixedtrw_feasible_at_maxiter()
test_fixedtrw_infeasible_subproblem()
