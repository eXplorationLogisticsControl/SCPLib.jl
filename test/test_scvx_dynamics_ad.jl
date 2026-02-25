"""Test problem with non-convex dynamics only, using AD"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParams_scvx_dynamics_ad
    μ::Float64
    u::Vector
    function ControlParams_scvx_dynamics_ad(μ::Float64)
        new(μ, zeros(4))
    end
end

function test_scvx_dynamics_ad(;verbosity::Int = 0)
    μ = 1.215058560962404e-02
    DU = 389703     # km
    TU = 382981     # sec
    MU = 500.0      # kg
    VU = DU/TU      # km/s
    params = ControlParams_scvx_dynamics_ad(μ)

    function eom!(drv, rv, p, t)
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
        r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
        drv[1:3] = rv[4:6]
        # derivatives of velocities
        drv[4] =  2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x);
        drv[5] = -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y;
        drv[6] = -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z;
        # append controls
        drv[4:6] += p.u[1:3]
        return
    end


    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]
    period_0 = 2.3538670417546639E+00

    rvf = [1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0]
    period_f = 3.3031221822879884

    # initial & final LPO
    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )

    # -------------------- create problem -------------------- #
    N = 60
    nx = 6
    nu = 4                              # [ux,uy,uz,Γ]
    tf = 2.6 
    times = LinRange(0.0, tf, N)

    thrust = 0.35    # N
    umax = thrust/MU/1e3 / (VU/TU)

    # create reference solution
    x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
    x_ref = zeros(nx,N)
    alphas = LinRange(0,1,N)
    for (i,alpha) in enumerate(alphas)
        x_ref[:,i] = (1-alpha)*x_along_lpo0[:,i] + alpha*x_along_lpof[:,i]
    end
    u_ref = zeros(nu, N-1)
    y_ref = nothing

    function objective(x, u)
        return sum(u[4,:])
    end

    # instantiate problem object    
    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        params,
        objective,
        times,
        x_ref,
        u_ref;
        ode_method = Vern7(),
    )
    set_silent(prob.model)

    # append boundary conditions
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == rv0)
    @constraint(prob.model, constraint_final_rv,   prob.model[:x][:,end] == rvf)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4,k] <= umax)

    # -------------------- instantiate algorithm -------------------- #
    tol_feas = 1e-6
    tol_opt = 1e-6
    algo = SCPLib.SCvx(nx, N; w = 1e3)

    # solve problem
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        tol_opt=tol_opt, tol_feas=tol_feas, verbosity = verbosity, maxiter = 100)

    # propagate solution
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @test maximum(abs.(g_dynamics_opt)) <= tol_feas
    @test solution.status == :Optimal
end


test_scvx_dynamics_ad(verbosity = verbosity)