"""Test for using scvxstar with custom propagate function"""

using BlockDiagonals
using Clarabel
using ForwardDiff
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


# -------------------- setup parameters -------------------- #
mutable struct ControlParams_custom_propagate_func
    μ::Float64
    u::Vector

    function ControlParams_custom_propagate_func(μ::Float64)
        new(μ, zeros(8))
    end
end


function test_scvxstar_custom_propagate_func(;verbosity::Int = 0, get_plot::Bool = false)
    # ----------------------------------------------------------------------------------- #
    # equations of motion
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
        return
    end


    function eom_aug!(dx_aug, x_aug, p, t)
        x, y, z = x_aug[1:3]
        vx, vy, vz = x_aug[4:6]

        r1vec = [x + p.μ, y, z]
        r2vec = [x - 1 + p.μ, y, z]
        r1 = norm(r1vec)
        r2 = norm(r2vec)

        dx_aug[1:3] = x_aug[4:6]
        # derivatives of velocities
        dx_aug[4] =  2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x);
        dx_aug[5] = -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y;
        dx_aug[6] = -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z;
        
        # Jacobian derivatives
        G1 = (1 - params.μ) / norm(r1vec)^5*(3*r1vec*r1vec' - norm(r1vec)^2*I(3))
        G2 = params.μ / norm(r2vec)^5*(3*r2vec*r2vec' - norm(r2vec)^2*I(3))
        Omega = [0 2 0; -2 0 0; 0 0 0]
        A = [zeros(3,3)                  I(3);
            G1 + G2 + diagm([1,1,0])    Omega]

        # derivatives of Phi_A, Phi_B
        dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
        return
    end


    function multi_spacecraft_eom!(drv, rv, p, t)
        @views for i in 1:N_spacecraft
            eom!(drv[1+6(i-1):6i], rv[1+6(i-1):6i], p, t)
        end
        return
    end


    function multi_spacecraft_eom_aug!(dx_aug, x_aug, p, t)
        nx = 6 * N_spacecraft
        nx2 = nx^2

        @views for i in 1:N_spacecraft
            eom!(dx_aug[1+6(i-1):6i], x_aug[1+6(i-1):6i], p, t)
            # dx[1+6(i-1):6i] = HighFidelityEphemerisModel.eom_NbodySH_Interp(x[1+6(i-1):6i], p, t)
        end
        
        Phi_aug = reshape(x_aug[nx+1:end], (nx,nx))'
        A_aug = zeros(nx,nx)
        for i in 1:N_spacecraft
            _x_copy = deepcopy(x_aug[1+6(i-1):6i])
            x, y, z = _x_copy[1:3]
            r1vec = [x + p.μ, y, z]
            r2vec = [x - 1 + p.μ, y, z]
            r1 = norm(r1vec)
            r2 = norm(r2vec)

            # Jacobian derivatives
            G1 = (1 - params.μ) / norm(r1vec)^5*(3*r1vec*r1vec' - norm(r1vec)^2*I(3))
            G2 = params.μ / norm(r2vec)^5*(3*r2vec*r2vec' - norm(r2vec)^2*I(3))
            Omega = [0 2 0; -2 0 0; 0 0 0]
            A = [zeros(3,3)                  I(3);
                G1 + G2 + diagm([1,1,0])    Omega]
            A_aug[1+6(i-1):6i,1+6(i-1):6i] = A
        end
        dx_aug[nx+1:end] = reshape((A_aug * Phi_aug)', nx2)   # julia is column-major
        return
    end


    # ----------------------------------------------------------------------------------- #
    # setup dynamics
    μ = 1.215058560962404e-02
    DU = 389703     # km
    TU = 382981     # sec
    MU = 500.0      # kg
    VU = DU/TU      # km/s
    params = ControlParams_custom_propagate_func(μ)

    N_spacecraft = 2

    # initial & final conditions
    period_0 = 2.3538670417546639E+00
    rv0 = [1.0809931218390707E+00, 0.0, -2.0235953267405354E-01, 0.0, -1.9895001215078018E-01, 0.0]
    rv1 = rv0 + [1e-2, 1e-2, 1e-3, 0.0, 0.0, 0.0]
    x0_concat = [rv0; rv1]

    period_f = 3.3031221822879884
    rvf = [1.1648780946517576, 0.0, -1.1145303634437023E-1, 0.0, -2.0191923237095796E-1, 0.0]
    xf_concat = [rvf; rvf]

    # initial & final LPO
    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )

    # ----------------------------------------------------------------------------------- #
    # test ODE propagation 
    dx_foo = zeros(12)
    multi_spacecraft_eom!(dx_foo, x0_concat, params, 0.0)

    # check ODE propagation
    ode_single_rv0 = ODEProblem(eom!, rv0, [0.0, π], params)
    sol_single_rv0 = solve(ode_single_rv0, Vern7(), reltol=1e-12, abstol=1e-12)

    ode_single_rv1 = ODEProblem(eom!, rv1, [0.0, π], params)
    sol_single_rv1 = solve(ode_single_rv1, Vern7(), reltol=1e-12, abstol=1e-12)

    ode_multiple = ODEProblem(multi_spacecraft_eom!, x0_concat, [0.0, π], params)
    sol_single_rvf = solve(ode_multiple, Vern7(), reltol=1e-12, abstol=1e-12)

    ode_multiple_aug = ODEProblem(multi_spacecraft_eom_aug!,
        [x0_concat; reshape(I(6*N_spacecraft), (6*N_spacecraft)^2)], [0.0, π], params)
    sol_aug = solve(ode_multiple_aug, Vern7(), reltol=1e-12, abstol=1e-12)

    @test maximum(abs.(sol_single_rvf.u[end][1:6] - sol_single_rv0.u[end])) < 1e-11
    @test maximum(abs.(sol_single_rvf.u[end][7:12] - sol_single_rv1.u[end])) < 1e-11


    # ----------------------------------------------------------------------------------- #
    # # custom ODESolution-like structure
    # mutable struct CustomODESolution
    #     u::Vector
    #     t::Vector
    # end

    # define custom propagation functions
    custom_get_trajectory = function (prob::SCPLib.ImpulsiveProblem, x_ref, u_ref, y_ref)
        sols = []
        g_dynamics = zeros(prob.nx, prob.N-1)
        for k in 1:size(x_ref,2)-1
            _x0_aug = [
                x_ref[:,k] + prob.dfdu(x_ref[:,k], u_ref[:,k], prob.times[k]) * u_ref[:,k];
                reshape(I(prob.nx), prob.nx^2)
            ]
            _ode = ODEProblem(multi_spacecraft_eom_aug!, _x0_aug, [prob.times[k], prob.times[k+1]], params)
            push!(sols, solve(_ode, Vern7(), reltol=1e-12, abstol=1e-12))
            
            g_dynamics[:,k] = x_ref[:,k+1] - sols[k].u[end][1:prob.nx]
        end
        return sols, g_dynamics
    end

    custom_set_dynamics_cache! = function (
        prob::SCPLib.ImpulsiveProblem,
        x_ref::Union{Matrix,Adjoint},
        u_ref::Union{Matrix,Adjoint},
        y_ref::Union{Matrix,Nothing},
    )
        # solve ensemble problem for each spacecraft & for each time-step
        sols, g_dynamics_ref = custom_get_trajectory(prob, x_ref, u_ref, y_ref)
        nx, N = size(x_ref)
        for (k,sol) in enumerate(sols)
            xf_aug = sol.u[end]
            prob.lincache.Φ_A[:,:,k] = reshape(xf_aug[nx+1:nx*(nx+1)], (nx,nx))'
            prob.lincache.Φ_B[:,:,k] = prob.lincache.Φ_A[:,:,k] * prob.dfdu(x_ref[:,k], u_ref[:,k], sol.t[1])
            prob.lincache.Φ_c[:,k]   = xf_aug[1:nx] - prob.lincache.Φ_A[:,:,k] * x_ref[:,k] - prob.lincache.Φ_B[:,:,k] * u_ref[:,k]
        end
        return g_dynamics_ref
    end

    # ----------------------------------------------------------------------------------- #
    # problem parameters
    N = 25
    nx = 6 * N_spacecraft
    nu = 4 * N_spacecraft                              # [ux,uy,uz,Γ]
    tf = 2.6 
    times = LinRange(0.0, tf, N)

    thrust = 0.35    # N
    umax = thrust/MU/1e3 / (VU/TU)

    function dfdu(x, u, t)
        return [BlockDiagonal([[zeros(3,3); I(3)], [zeros(3,3); I(3)]]) zeros(12,2)]
    end

    # initial reference solution
    x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
    x_ref = zeros(nx,N)
    alphas = LinRange(0,1,N)
    for (i,alpha) in enumerate(alphas)
        x_ref[1:6,i] = (1-alpha)*x_along_lpo0[:,i] + alpha*x_along_lpof[:,i]
        x_ref[7:12,i] = x_ref[1:6,i]
    end
    u_ref = zeros(nu, N)
    y_ref = nothing

    # objective function
    function objective(x, u, y)
        return sum(u[7,:]) + sum(u[8,:])
    end

    # construct problem struct
    prob = SCPLib.ImpulsiveProblem(
        Clarabel.Optimizer,
        multi_spacecraft_eom!,
        params,
        objective,
        times,
        x_ref,
        u_ref,
        y_ref;
        dfdu = dfdu,
        eom_aug! = multi_spacecraft_eom_aug!,
        ode_ensemble_method = EnsembleSerial(),
        ode_method = Vern7(),
        ode_reltol = 1e-12,
        ode_abstol = 1e-12,
        fun_get_trajectory = custom_get_trajectory,
        set_dynamics_cache! = custom_set_dynamics_cache!,
    )
    set_silent(prob.model)

    # append boundary conditions
    @constraint(prob.model, constraint_initial_x, prob.model[:x][:,1] == x0_concat)
    @constraint(prob.model, constraint_final_x, prob.model[:x][:,end] + dfdu(0,0,0) * prob.model[:u][:,end] == xf_concat)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control_sc1[k in 1:N],
        [prob.model[:u][7,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_associate_control_sc2[k in 1:N],
        [prob.model[:u][8,k], prob.model[:u][4:6,k]...] in SecondOrderCone())

    @constraint(prob.model, constraint_control_magnitude_sc1[k in 1:N],
        prob.model[:u][7,k] <= umax)
    @constraint(prob.model, constraint_control_magnitude_sc2[k in 1:N],
        prob.model[:u][8,k] <= umax)


    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(nx, N; w0 = 1e4)

    # solve problem
    tol_opt = 1e-6
    tol_feas = 1e-8
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref, y_ref; 
        verbosity = verbosity, maxiter = 100, tol_opt = tol_opt, tol_feas = tol_feas)

    # propagate solution
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u, solution.y)
    @test maximum(abs.(g_dynamics_opt)) <= tol_feas
    @test solution.status == :Optimal

    # -------------------- plot -------------------- #
    if get_plot
        fig = Figure(size=(800, 500))
        ax3d = Axis3(fig[1,1]; aspect=:equal, xlabel="x", ylabel="y", zlabel="z")
        for (isol, _sol) in enumerate(sols_opt)
            lines!(ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=:black)
            scatter!(ax3d, Array(_sol)[1,1], Array(_sol)[2,1], Array(_sol)[3,1], color=:black)
            scatter!(ax3d, Array(_sol)[1,end], Array(_sol)[2,end], Array(_sol)[3,end], color=:black)

            lines!(ax3d, Array(_sol)[7,:], Array(_sol)[8,:], Array(_sol)[9,:], color=:deeppink)
            scatter!(ax3d, Array(_sol)[7,1], Array(_sol)[8,1], Array(_sol)[9,1], color=:deeppink)
            scatter!(ax3d, Array(_sol)[7,end], Array(_sol)[8,end], Array(_sol)[9,end], color=:deeppink)
        end
        lines!(ax3d, Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:blue)
        lines!(ax3d, Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:green)
        
        axu = Axis(fig[1,2], xlabel="Time", ylabel="Control magnitude")
        stem!(axu, times, solution.u[7,:], color=:black)
        stem!(axu, times, solution.u[8,:], color=:deeppink)
        display(fig)
    end
end

test_scvxstar_custom_propagate_func(verbosity = verbosity, get_plot=get_plot)