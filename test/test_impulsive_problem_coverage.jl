"""Exercise the `ImpulsiveProblem` construction paths the physical test problems never take"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


impulsive_coverage_dfdu(x, u, t) = reshape([0.0, 1.0], 2, 1)
impulsive_coverage_g_noncvx(cache, x, u) = [x[1,2]^2 - 0.25]
impulsive_coverage_h_noncvx(cache, x, u) = [sum(u.^2) - 1.0]


"""
Drifting double integrator whose control is a velocity impulse at each node.

Neither `eom_aug!` nor the constraint Jacobians are supplied, so the constructor
builds them by automatic differentiation.
"""
function make_impulsive_coverage_problem()
    nx, nu, N, Nu = 2, 1, 3, 2
    x_ref = [0.0 0.5 1.0;
             1.0 1.0 1.0]
    u_ref = [0.1 0.2]

    # `get_impulsive_augmented_eom` hands the full augmented vector to `eom!`,
    # so only the first `nx` entries may be touched
    function eom!(dx, x, pu, t)
        dx[1] = x[2]
        dx[2] = 0.0
        return
    end

    prob = SCPLib.ImpulsiveProblem(
        Clarabel.Optimizer,
        eom!,
        nothing,
        (x, u) -> sum(u.^2),
        LinRange(0.0, 1.0, N),
        x_ref,
        u_ref;
        dfdu = impulsive_coverage_dfdu,
        ng = 1,
        g_noncvx = impulsive_coverage_g_noncvx,
        nh = 1,
        h_noncvx = impulsive_coverage_h_noncvx,
    )
    set_silent(prob.model)
    return prob, x_ref, u_ref
end


function test_impulsive_noncvx_variables_are_registered()
    prob, _, _ = make_impulsive_coverage_problem()

    @test haskey(object_dictionary(prob.model), :ξ)
    @test haskey(object_dictionary(prob.model), :ζ)
    @test :constraint_g_noncvx in prob.model_nl_references
    @test :constraint_h_noncvx in prob.model_nl_references
end


function test_impulsive_flatten_roundtrip()
    prob, x_ref, u_ref = make_impulsive_coverage_problem()

    z = SCPLib.stack_flatten_variables(prob, x_ref, u_ref)
    @test length(z) == prob.nx * prob.N + prob.nu * prob.Nu

    x_back, u_back = SCPLib.unpack_flattened_variables(prob, z)
    @test x_back == x_ref
    @test u_back == u_ref
end


function test_impulsive_autodiff_constraint_jacobians()
    prob, x_ref, u_ref = make_impulsive_coverage_problem()

    _, g_ref, h_ref = SCPLib.set_linearized_constraints!(prob, x_ref, u_ref)

    @test g_ref ≈ [x_ref[1,2]^2 - 0.25]
    @test h_ref ≈ max.([sum(u_ref.^2) - 1.0], 0)

    # z = [x11, x21, x12, x22, x13, x23, u11, u12]
    ∇g_expected = zeros(1, 8)
    ∇g_expected[1,3] = 2 * x_ref[1,2]
    @test prob.lincache.∇g ≈ ∇g_expected

    ∇h_expected = zeros(1, 8)
    ∇h_expected[1,7:8] = 2 * vec(u_ref)
    @test prob.lincache.∇h ≈ ∇h_expected
end


test_impulsive_noncvx_variables_are_registered()
test_impulsive_flatten_roundtrip()
test_impulsive_autodiff_constraint_jacobians()
