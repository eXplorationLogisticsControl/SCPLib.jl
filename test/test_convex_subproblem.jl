"""Dev for continuous problem"""

using Clarabel
using GLMakie
using JuMP
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end

function test_convex_subproblem()
    
    μ = 1.215058560962404e-02
    params = ControlParams(μ)

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


    function eom(rv, p, t)
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
        r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
        drv = [
            vx; vy; vz;
            2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x) + p.u[1];
            -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y + p.u[2];
            -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z + p.u[3];
        ]
        return drv
    end


    # cache for Jacobian
    sd = SymbolicsSparsityDetection()
    adtype = AutoSparse(AutoFiniteDiff())
    cache_jac_A = sparse_jacobian_cache(adtype, sd, (y,x) -> eom!(y, x, params, 1.0), zeros(6), zeros(6))

    function eom_aug!(dx_aug, x_aug, p, t)
        # compute state derivative with control
        # eom!(dx_aug[1:6], x_aug[1:6], p, t)
        dx_aug[1:6] = eom(x_aug[1:6], p, t)
        
        # ForwardDiff.jacobian!(A, (y,x) -> eom!(y, x, p, t), dx_aug[1:6], x_aug[1:6])
        A = ForwardDiff.jacobian(x -> eom(x, p, t), x_aug[1:6])
        # A = cr3bp_dfdx(x_aug[1:6], p)
        #, cfg::JacobianConfig = JacobianConfig(eom!, dx_aug[1:6], x_aug[1:6]), check=Val{true}())
        # A = ForwardDiff.jacobian(x -> eom!(zeros(6), x, params, t), x_aug[1:6])

        # A = sparse_jacobian(adtype, cache_jac_A, (y,x) -> eom!(y, x, p, t), dx_aug[1:6], x_aug[1:6])
        # A = sparse_jacobian(adtype, cache_jac_A, (y,x) -> eom!(y, x, p, t), similar(x_aug[1:6]), x_aug[1:6])
        B = [zeros(3,4); I(3) zeros(3,1)]
        
        # extract STMs (note: julia is column-major)
        #Phi_A = reshape(x_aug[nx+1:nx*(nx+1)], (nx,nx))'
        Phi_B = reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))'  # note: julia is column-major

        # derivatives of Phi_A, Phi_B
        dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * Phi_B + B)', nx*nu)
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
    umax = 0.05                         # max control magnitudes

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

    # plot initial guess
    fig = Figure()
    ax3d = Axis3(fig[1,1]; aspect=:data)
    lines!(Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:blue)
    lines!(Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:green)
    # scatter!(x_ref[1,:], x_ref[2,:], x_ref[3,:], color=:black)

    function objective(x, u, y)
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
        u_ref,
        y_ref;
        eom_aug! = eom_aug!,
    )

    # append boundary conditions
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == rv0)
    @constraint(prob.model, constraint_final_rv,   prob.model[:x][:,end] == rvf)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4,k] <= umax)

    # propagate initial guess
    sols_ig, g_dynamics_ig = SCPLib.get_trajectory(prob, x_ref, u_ref, y_ref)
    for _sol in sols_ig
        lines!(ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=:black)
    end

    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(
        nx, N;
    )

    SCPLib.set_noncvx_expressions!(prob, x_ref, u_ref, y_ref)
    SCPLib.set_trust_region_constraints!(algo, prob, 0.5, x_ref, u_ref)

    # try solving subproblem
    SCPLib.solve_convex_subproblem!(algo, prob)
    @test termination_status(prob.model) == OPTIMAL
end