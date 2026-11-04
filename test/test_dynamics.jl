"""Compare generated `eom_aug!` helpers against a user-defined reference"""

using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


struct DynamicsTestParams
    μ::Float64
end


function cr3bp_eom!(drv, rv, pu, t)
    (; params, u) = pu
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt((x + params.μ)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + params.μ)^2 + y^2 + z^2)
    drv[1:3] = rv[4:6]
    drv[4] =  2*vy + x - ((1 - params.μ)/r1^3)*(params.μ + x) + (params.μ/r2^3)*(1 - params.μ - x)
    drv[5] = -2*vx + y - ((1 - params.μ)/r1^3)*y - (params.μ/r2^3)*y
    drv[6] = -((1 - params.μ)/r1^3)*z - (params.μ/r2^3)*z
    drv[4:6] += u[1:3]
    return
end


function cr3bp_dfdx(xstate, u, params, t)
    r1vec = [xstate[1] + params.μ, xstate[2], xstate[3]]
    r2vec = [xstate[1] - 1 + params.μ, xstate[2], xstate[3]]
    G1 = (1 - params.μ) / norm(r1vec)^5 * (3*r1vec*r1vec' - norm(r1vec)^2*I(3))
    G2 = params.μ / norm(r2vec)^5 * (3*r2vec*r2vec' - norm(r2vec)^2*I(3))
    Omega = [0.0 2.0 0.0; -2.0 0.0 0.0; 0.0 0.0 0.0]
    return [zeros(3, 3) I(3);
            G1 + G2 + diagm([1.0, 1.0, 0.0]) Omega]
end


function cr3bp_dfdu(xstate, u, params, t)
    return [zeros(3, 4); I(3) zeros(3, 1)]
end


"""User-defined continuous augmented EOM, matching test_scvxstar_dynamics_userdefined.jl"""
function cr3bp_eom_aug!(dx_aug, x_aug, pu, t)
    (; params, u) = pu
    nx, nu = 6, 4
    x, y, z = x_aug[1:3]
    vx, vy, vz = x_aug[4:6]

    r1vec = [x + params.μ, y, z]
    r2vec = [x - 1 + params.μ, y, z]
    r1 = norm(r1vec)
    r2 = norm(r2vec)

    dx_aug[1:3] = x_aug[4:6]
    dx_aug[4] =  2*vy + x - ((1 - params.μ)/r1^3)*(params.μ + x) + (params.μ/r2^3)*(1 - params.μ - x)
    dx_aug[5] = -2*vx + y - ((1 - params.μ)/r1^3)*y - (params.μ/r2^3)*y
    dx_aug[6] = -((1 - params.μ)/r1^3)*z - (params.μ/r2^3)*z
    dx_aug[4:6] += u[1:3]

    G1 = (1 - params.μ) / norm(r1vec)^5 * (3*r1vec*r1vec' - norm(r1vec)^2*I(3))
    G2 = params.μ / norm(r2vec)^5 * (3*r2vec*r2vec' - norm(r2vec)^2*I(3))
    Omega = [0.0 2.0 0.0; -2.0 0.0 0.0; 0.0 0.0 0.0]
    A = [zeros(3, 3) I(3);
         G1 + G2 + diagm([1.0, 1.0, 0.0]) Omega]
    B = [zeros(3, 4); I(3) zeros(3, 1)]

    dx_aug[7:42] = reshape(A * reshape(x_aug[7:42], 6, 6), 36)
    dx_aug[nx*(nx + 1)+1:nx*(nx + 1)+nx*nu] = reshape(A * reshape(x_aug[nx*(nx + 1)+1:nx*(nx + 1)+nx*nu], (nx, nu)) + B, nx*nu)
    return
end


"""User-defined impulsive augmented EOM (state + Φ_A only)"""
function cr3bp_impulsive_eom_aug!(dx_aug, x_aug, pu, t)
    cr3bp_eom!(dx_aug, x_aug, pu, t)
    A = cr3bp_dfdx(x_aug[1:6], pu.u, pu.params, t)
    dx_aug[7:42] = reshape(A * reshape(x_aug[7:42], 6, 6), 36)
    return
end


function test_continuous_generated_eom_aug_matches_userdefined()
    nx, nu = 6, 4
    params = DynamicsTestParams(1.215058560962404e-2)
    x = [1.0809931218390707, 0.0, -0.20235953267405354,
         0.0, -0.19895001215078018, 0.0]
    u = [0.01, -0.02, 0.03, 0.04]
    p = SCPLib.dynamics_input(params, u)

    Random.seed!(1)
    Φ_A = Matrix(1.0*I(nx)) + 0.1 * randn(nx, nx)
    Φ_B = 0.1 * randn(nx, nu)
    x_aug = [x; vec(Φ_A); vec(Φ_B)]

    eom_aug_jac! = SCPLib.get_continuous_augmented_eom(cr3bp_eom!, cr3bp_dfdx, cr3bp_dfdu, nx, nu)
    eom_aug_ad!  = SCPLib.get_continuous_augmented_eom(cr3bp_eom!, params, nx, nu)

    dx_user = zeros(nx + nx^2 + nx*nu)
    dx_jac  = zeros(nx + nx^2 + nx*nu)
    dx_ad   = zeros(nx + nx^2 + nx*nu)
    cr3bp_eom_aug!(dx_user, x_aug, p, 0.0)
    eom_aug_jac!(dx_jac, x_aug, p, 0.0)
    eom_aug_ad!(dx_ad, x_aug, p, 0.0)

    @test dx_jac ≈ dx_user atol=1e-12
    @test dx_ad  ≈ dx_user atol=1e-10
end


function test_impulsive_generated_eom_aug_matches_userdefined()
    nx = 6
    params = DynamicsTestParams(1.215058560962404e-2)
    x = [1.0809931218390707, 0.0, -0.20235953267405354,
         0.0, -0.19895001215078018, 0.0]
    u = [0.01, -0.02, 0.03, 0.04]
    p = SCPLib.dynamics_input(params, u)

    Random.seed!(2)
    Φ_A = Matrix(1.0*I(nx)) + 0.1 * randn(nx, nx)
    x_aug = [x; vec(Φ_A)]

    eom_aug_jac! = SCPLib.get_impulsive_augmented_eom(cr3bp_eom!, cr3bp_dfdx, nx)
    eom_aug_ad!  = SCPLib.get_impulsive_augmented_eom(cr3bp_eom!, params, nx)

    dx_user = zeros(nx + nx^2)
    dx_jac  = zeros(nx + nx^2)
    dx_ad   = zeros(nx + nx^2)
    cr3bp_impulsive_eom_aug!(dx_user, x_aug, p, 0.0)
    eom_aug_jac!(dx_jac, x_aug, p, 0.0)
    eom_aug_ad!(dx_ad, x_aug, p, 0.0)

    @test dx_jac ≈ dx_user atol=1e-12
    @test dx_ad  ≈ dx_user atol=1e-10
end


test_continuous_generated_eom_aug_matches_userdefined()
test_impulsive_generated_eom_aug_matches_userdefined()
