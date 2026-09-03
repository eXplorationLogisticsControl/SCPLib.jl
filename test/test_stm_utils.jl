"""Test STM via ForwardDiff through ODE vs augmented EOM propagation"""

using DiffResults
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end
if !@isdefined verbosity
    verbosity = 0
end


struct ControlParams_stm_utils
    μ::Float64
end


function test_stm_x_adode_vs_augmented_eom(;verbosity::Int = 0)
    μ = 1.215058560962404e-02
    params = ControlParams_stm_utils(μ)
    nx = 6
    tspan = (0.0, 2.3538670417546639E+00)
    u = zeros(4)
    p = SCPLib.dynamics_input(params, u)

    function eom!(drv, rv, pu, t)
        (; params, u) = pu
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+params.μ)^2 + y^2 + z^2 )
        r2 = sqrt( (x-1+params.μ)^2 + y^2 + z^2 )
        drv[1:3] = rv[4:6]
        drv[4] =  2*vy + x - ((1-params.μ)/r1^3)*(params.μ+x) + (params.μ/r2^3)*(1-params.μ-x)
        drv[5] = -2*vx + y - ((1-params.μ)/r1^3)*y - (params.μ/r2^3)*y
        drv[6] = -((1-params.μ)/r1^3)*z - (params.μ/r2^3)*z
        drv[4:6] += u[1:3]
        return
    end

    function f_dfdx(x, u, params, t)
        return ForwardDiff.jacobian(
            (y, x_) -> eom!(y, x_, SCPLib.dynamics_input(params, u), t), zeros(nx), x)
    end

    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]

    ode_kwargs = (; reltol = 1e-12, abstol = 1e-12)

    t_ad = @elapsed begin
        result = SCPLib.get_stm_x_adode(eom!, rv0, tspan, p, Vern7(); ode_kwargs...)
    end
    xf_ad = DiffResults.value(result)
    Φ_ad = DiffResults.jacobian(result)

    t_aug = @elapsed begin
        eom_aug! = SCPLib.get_impulsive_augmented_eom(eom!, f_dfdx, nx)
        x0_aug = [rv0; reshape(Matrix{Float64}(I, nx, nx), nx^2)]
        sol_aug = solve(ODEProblem(eom_aug!, x0_aug, tspan, p), Vern7(); ode_kwargs...)
    end
    xf_aug = sol_aug.u[end][1:nx]
    Φ_aug = reshape(sol_aug.u[end][nx+1:nx*(nx+1)], nx, nx)

    if verbosity == 1
        println("STM Φ_x  ForwardDiff:  $(t_ad) s")
        println("STM Φ_x  augmented:    $(t_aug) s")
    end

    @test xf_ad ≈ xf_aug atol=1e-10
    @test Φ_ad ≈ Φ_aug atol=1e-8
end


function test_stm_u_adode_vs_augmented_eom(;verbosity::Int = 0)
    μ = 1.215058560962404e-02
    params = ControlParams_stm_utils(μ)
    nx = 6
    nu = 4
    tspan = (0.0, 2.3538670417546639E+00)
    u = [1.0e-3, -2.0e-3, 5.0e-4, sqrt(1e-3^2 + 2e-3^2 + 5e-4^2)]
    p = SCPLib.dynamics_input(params, u)

    function eom!(drv, rv, pu, t)
        (; params, u) = pu
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+params.μ)^2 + y^2 + z^2 )
        r2 = sqrt( (x-1+params.μ)^2 + y^2 + z^2 )
        drv[1:3] = rv[4:6]
        drv[4] =  2*vy + x - ((1-params.μ)/r1^3)*(params.μ+x) + (params.μ/r2^3)*(1-params.μ-x)
        drv[5] = -2*vx + y - ((1-params.μ)/r1^3)*y - (params.μ/r2^3)*y
        drv[6] = -((1-params.μ)/r1^3)*z - (params.μ/r2^3)*z
        drv[4:6] += u[1:3]
        return
    end

    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]

    ode_kwargs = (; reltol = 1e-12, abstol = 1e-12)

    t_ad = @elapsed begin
        result = SCPLib.get_stm_u_adode(eom!, rv0, u, tspan, p, Vern7(); ode_kwargs...)
    end
    xf_ad = DiffResults.value(result)
    Φ_B_ad = DiffResults.jacobian(result)

    t_aug = @elapsed begin
        eom_aug! = SCPLib.get_continuous_augmented_eom(eom!, params, nx, nu)
        x0_aug = SCPLib.init_continuous_dynamics_xaug(rv0, nx, nu)
        sol_aug = solve(ODEProblem(eom_aug!, x0_aug, tspan, p), Vern7(); ode_kwargs...)
    end
    xf_aug = sol_aug.u[end][1:nx]
    Φ_B_aug = reshape(sol_aug.u[end][nx*(nx+1)+1:nx*(nx+1)+nx*nu], nx, nu)

    if verbosity == 1
        println("STM Φ_u  ForwardDiff:  $(t_ad) s")
        println("STM Φ_u  augmented:    $(t_aug) s")
    end

    @test xf_ad ≈ xf_aug atol=1e-10
    @test Φ_B_ad ≈ Φ_B_aug atol=1e-8
end

function test_stm_adode_vs_augmented_eom(;verbosity::Int = 0)
    μ = 1.215058560962404e-02
    params = ControlParams_stm_utils(μ)
    nx = 6
    nu = 4
    tspan = (0.0, 2.3538670417546639E+00)
    u = [1.0e-3, -2.0e-3, 5.0e-4, sqrt(1e-3^2 + 2e-3^2 + 5e-4^2)]
    p = SCPLib.dynamics_input(params, u)

    function eom!(drv, rv, pu, t)
        (; params, u) = pu
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+params.μ)^2 + y^2 + z^2 )
        r2 = sqrt( (x-1+params.μ)^2 + y^2 + z^2 )
        drv[1:3] = rv[4:6]
        drv[4] =  2*vy + x - ((1-params.μ)/r1^3)*(params.μ+x) + (params.μ/r2^3)*(1-params.μ-x)
        drv[5] = -2*vx + y - ((1-params.μ)/r1^3)*y - (params.μ/r2^3)*y
        drv[6] = -((1-params.μ)/r1^3)*z - (params.μ/r2^3)*z
        drv[4:6] += u[1:3]
        return
    end

    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]

    ode_kwargs = (; reltol = 1e-12, abstol = 1e-12)

    t_ad = @elapsed begin
        result = SCPLib.get_stm_adode(eom!, rv0, u, tspan, p, Vern7(); ode_kwargs...)
    end
    xf_ad = DiffResults.value(result)
    Φ_ad = DiffResults.jacobian(result)
    Φ_A_ad = Φ_ad[:, 1:nx]
    Φ_B_ad = Φ_ad[:, nx+1:end]

    t_aug = @elapsed begin
        eom_aug! = SCPLib.get_continuous_augmented_eom(eom!, params, nx, nu)
        x0_aug = SCPLib.init_continuous_dynamics_xaug(rv0, nx, nu)
        sol_aug = solve(ODEProblem(eom_aug!, x0_aug, tspan, p), Vern7(); ode_kwargs...)
    end
    xf_aug = sol_aug.u[end][1:nx]
    Φ_A_aug = reshape(sol_aug.u[end][nx+1:nx*(nx+1)], nx, nx)
    Φ_B_aug = reshape(sol_aug.u[end][nx*(nx+1)+1:nx*(nx+1)+nx*nu], nx, nu)

    if verbosity == 1
        println("STM Φ_xu ForwardDiff:  $(t_ad) s")
        println("STM Φ_xu augmented:    $(t_aug) s")
    end

    @test xf_ad ≈ xf_aug atol=1e-10
    @test Φ_A_ad ≈ Φ_A_aug atol=1e-8
    @test Φ_B_ad ≈ Φ_B_aug atol=1e-8
end

test_stm_x_adode_vs_augmented_eom(;verbosity=verbosity)
test_stm_u_adode_vs_augmented_eom(;verbosity=verbosity)
test_stm_adode_vs_augmented_eom(;verbosity=verbosity)
