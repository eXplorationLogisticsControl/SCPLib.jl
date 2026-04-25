"""Forward-backward shooting test"""

using Clarabel
using ForwardDiff
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

seed = 1234
Random.seed!(seed)

# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParamsForwardBackward
    μ::Float64
    u::Vector
    function ControlParamsForwardBackward(μ::Float64)
        new(μ, zeros(4))
    end
end

function test_scvxstar_forwardbackward(;verbosity::Int = 0)
    # define dynamics constants
    μ = 1.215058560962404e-02
    DU = 389703     # km
    TU = 382981     # sec
    MU = 500.0      # kg
    VU = DU/TU      # km/s
    params = ControlParamsForwardBackward(μ)

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

    function eom_aug!(dx_aug, x_aug, p, t)
        # state derivatives
        eom!(view(dx_aug, 1:6), x_aug[1:6], p, t)
        
        # STM derivatives
        r1vec = [x_aug[1] + p.μ, x_aug[2], x_aug[3]]
        r2vec = [x_aug[1] - 1 + p.μ, x_aug[2], x_aug[3]]
        G1 = (1 - params.μ) / norm(r1vec)^5*(3*r1vec*r1vec' - norm(r1vec)^2*I(3))
        G2 = params.μ / norm(r2vec)^5*(3*r2vec*r2vec' - norm(r2vec)^2*I(3))
        Omega = [0 2 0; -2 0 0; 0 0 0]
        A = [zeros(3,3)                  I(3);
            G1 + G2 + diagm([1,1,0])    Omega]
        B = [zeros(3,4); I(3) zeros(3,1)]

        # derivatives of Phi_A, Phi_B
        dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
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

    # -------------------- create problem -------------------- #
    N = 20
    nx = 6
    nu = 4                              # [ux,uy,uz,Γ]
    tf = 2.6 
    times = LinRange(0.0, tf, N)

    thrust = 0.35    # N
    umax = thrust/MU/1e3 / (VU/TU)

    # initial & final LPO
    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )

    # create reference solution
    x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
    x_ref = [rv0 rvf]

    # control initial guess
    u_ref = zeros(nu, N-1)
    u_ref[1:3,:] = randn(3, N-1) * umax * 0.1
    u_ref[4,:] = norm.(eachcol(u_ref[1:3,:]))

    # instantiate problem object    
    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        params,
        (x,u) -> sum(u[4,:]),
        times,
        x_ref,
        u_ref;
        shooting_method = :forwardbackward,
        ode_method = Vern7(),
    )

    # evaluate nonlinear constraints
    sols_ref, g_dynamics_ref = SCPLib.get_trajectory_augmented_forwardbackward(prob, x_ref, u_ref)

    # ------------------------------------------------------------------------------------------------------------------------ #
    # evaluate nonlinear constraints Jacobian analytically
    Φ_A_list, Φ_B_list = SCPLib.set_continuous_dynamics_cache!(prob.lincache, x_ref, u_ref, sols_ref)
    # @show prob.lincache.∇g_dyn

    # ------------------------------------------------------------------------------------------------------------------------ #
    # evaluate nonlinear constraints Jacobian via finite difference
    ∇g_dyn_fd = zeros(nx, 2nx + nu*(N-1))
    for i in 1:6
        x0_ptrb_pls = deepcopy(x_ref)
        x0_ptrb_pls[i,1] += 1e-6

        x0_ptrb_min = deepcopy(x_ref)
        x0_ptrb_min[i,1] -= 1e-6
        _, _g_pls = SCPLib.get_trajectory_augmented_forwardbackward(prob, x0_ptrb_pls, u_ref)
        _, _g_min = SCPLib.get_trajectory_augmented_forwardbackward(prob, x0_ptrb_min, u_ref)

        ∇g_dyn_fd[:,i] = -(_g_pls - _g_min) / (2e-6)    # forward segment has minus sign
    end

    for i in 1:6
        x0_ptrb_pls = deepcopy(x_ref)
        x0_ptrb_pls[i,2] += 1e-6

        x0_ptrb_min = deepcopy(x_ref)
        x0_ptrb_min[i,2] -= 1e-6
        _, _g_pls = SCPLib.get_trajectory_augmented_forwardbackward(prob, x0_ptrb_pls, u_ref)
        _, _g_min = SCPLib.get_trajectory_augmented_forwardbackward(prob, x0_ptrb_min, u_ref)

        ∇g_dyn_fd[:,6+i] = (_g_pls - _g_min) / (2e-6)    # backward segment has plus sign
    end

    for i in 1:nu*(N-1)
        u_ptrb_pls = deepcopy(u_ref)
        u_ptrb_pls[i] += 1e-6

        u_ptrb_min = deepcopy(u_ref)
        u_ptrb_min[i] -= 1e-6
        _, _g_pls = SCPLib.get_trajectory_augmented_forwardbackward(prob, x_ref, u_ptrb_pls)
        _, _g_min = SCPLib.get_trajectory_augmented_forwardbackward(prob, x_ref, u_ptrb_min)

        ∇g_dyn_fd[:,2nx+i] = (_g_pls - _g_min) / (2e-6)    # backward segment has plus sign
    end

    # error in gradients
    err = prob.lincache.∇g_dyn - ∇g_dyn_fd
    @test maximum(err) < 1e-6


    # ------------------------------------------------------------------------------------------------------------------------ #
    # append convex constraints to the model
    # append boundary conditions
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == rv0)
    @constraint(prob.model, constraint_final_rv,   prob.model[:x][:,end] == rvf)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4,k] <= umax)

    set_silent(prob.model)

    # solve
    tol_opt = 1e-6
    tol_feas = 1e-6
    algo = SCPLib.SCvxStar(nx, N; w0 = 1e2, shooting_method = :forwardbackward)

    # solve problem
    u_ref = zeros(nu, N-1)
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref; tol_opt=tol_opt, tol_feas=tol_feas, maxiter = 100, verbosity=verbosity)

    # evaluate nonlinear constraints
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory_augmented_forwardbackward(prob, solution.x, solution.u)
    @test maximum(abs.(g_dynamics_opt)) <= 1e-6
    @test solution.status == :Optimal
end


test_scvxstar_forwardbackward(verbosity = verbosity)