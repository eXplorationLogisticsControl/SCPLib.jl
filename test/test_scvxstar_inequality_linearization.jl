"""Test linearization of non-convex inequality constraints and the SCvx* inequality multiplier update"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end
if !@isdefined verbosity
    verbosity = 0
end


function _ineqlin_eom!(dx, x, pu, t)
    dx[1] = pu.u[1]
    return
end


"""
1-state toy problem: dx = u on t ∈ [0,1], x₁ = 0, maximize x₂ s.t. x₂ ≤ 1 (convex)
and the non-convex inequality `h_noncvx`.
"""
function _make_ineqlin_problem(h_noncvx; x_ref = zeros(1, 2), u_ref = zeros(1, 1), x2_lb = nothing)
    x_ref = copy(x_ref)
    u_ref = copy(u_ref)
    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        _ineqlin_eom!,
        nothing,
        (x, u) -> -x[1, 2],
        [0.0, 1.0],
        x_ref,
        u_ref;
        nh = 1,
        h_noncvx = h_noncvx,
    )
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][1, 1] == 0.0)
    @constraint(prob.model, prob.model[:x][1, 2] <= 1.0)
    if !isnothing(x2_lb)
        @constraint(prob.model, prob.model[:x][1, 2] >= x2_lb)
    end
    return prob, x_ref, u_ref
end


function test_inequality_linearization_uses_raw_h()
    # h = x₂ - 2 is strictly inactive (h = -2) at the reference x₂ = 0
    prob, x_ref, u_ref = _make_ineqlin_problem((c, x, u) -> [x[1, 2] - 2.0])
    _, _, h_ref = SCPLib.set_linearized_constraints!(prob, x_ref, u_ref)

    # returned h_ref is h⁺ (used only in the merit function)
    @test h_ref == [0.0]

    # model constraint: h(z̄) + ∇h Δz <= ζ, i.e. -2 + x₂ - ζ <= 0.
    # JuMP moves the constant to the right-hand side: x₂ - ζ <= 2 = -h(z̄)
    con = prob.model[:constraint_h_noncvx][1]
    @test normalized_rhs(con) ≈ 2.0
    @test normalized_coefficient(con, prob.model[:x][1, 2]) ≈ 1.0
    @test normalized_coefficient(con, prob.model[:ζ][1]) ≈ -1.0
    @test constraint_object(con).set == MOI.LessThan(2.0)
end


function test_inequality_linearization_inactive_no_barrier(; verbosity::Int = 0)
    h_inactive = (c, x, u) -> [x[1, 2] - 2.0]   # never active since x₂ <= 1 convexly

    # SCvx* with fixed and auto-tuned initial weight
    for w0 in (10.0, nothing)
        prob, x_ref, u_ref = _make_ineqlin_problem(h_inactive)
        algo = SCPLib.SCvxStar(1, 2; nh = 1, w0 = w0)
        sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 200)
        @test sol.status == :Optimal
        @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
    end

    # SCvx* with l1_penalty (NormOneCone epigraph on ζ) and auto-tuned initial weight
    prob, x_ref, u_ref = _make_ineqlin_problem(h_inactive)
    algo = SCPLib.SCvxStar(1, 2; nh = 1, w0 = nothing, l1_penalty = true)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 200)
    @test sol.status == :Optimal
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6

    # SCvx
    prob, x_ref, u_ref = _make_ineqlin_problem(h_inactive)
    algo = SCPLib.SCvx(1, 2; w = 1e3)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 200)
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6

    # ProxLinear
    prob, x_ref, u_ref = _make_ineqlin_problem(h_inactive)
    algo = SCPLib.ProxLinear(1e2, 1.0)
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 200)
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
end


function test_scvxstar_inequality_multiplier_update(; verbosity::Int = 0)
    # h = x₂² - 4 is violated at the initial guess x₂ = -3 and inactive at the optimum x₂ = 1
    h = (c, x, u) -> [x[1, 2]^2 - 4.0]
    x_ref0 = [0.0 -3.0]
    u_ref0 = fill(-3.0, 1, 1)
    prob, x_ref, u_ref = _make_ineqlin_problem(h; x_ref = x_ref0, u_ref = u_ref0, x2_lb = -10.0)
    # w0 = 10 so that the stationarity-gated multiplier update (|ΔJ| < δ) fires at an
    # accepted iterate where h < 0; with w0 = 1 it only fires once (at the violated first
    # iterate) before convergence, which would not exercise the decrease of μ.
    algo = SCPLib.SCvxStar(1, 2; nh = 1, w0 = 10.0)
    μhist = Float64[]
    sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = verbosity, maxiter = 200,
        callback = (a, s, i, J, c) -> push!(μhist, a.μ[1]))

    h_final = sol.x[1, 2]^2 - 4.0
    @test sol.status == :Optimal
    @test sol.x[1, 2] ≈ 1.0 atol = 1e-6
    @test h_final < 0
    @test algo.μ[1] ≈ 0.0 atol = 1e-8
    @test abs(algo.μ[1] * h_final) <= 1e-8          # complementarity
    @test all(μhist .>= 0.0)
    @test maximum(μhist) > algo.μ[1]                # μ grew while violated, then decreased
end


@testset "Inequality linearization" begin
    test_inequality_linearization_uses_raw_h()
    test_inequality_linearization_inactive_no_barrier(verbosity = verbosity)
    test_scvxstar_inequality_multiplier_update(verbosity = verbosity)
end
