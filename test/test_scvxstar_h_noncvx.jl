"""Test with SCvx* that includes h_noncvx"""

using Clarabel
using ForwardDiff
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

# ODE parameters
mutable struct QuadroptorParams
    u::Vector
end


function test_scvxstar_h_noncvx(;verbosity::Int = 0)
    # -------------------- setup problem -------------------- #
    # system parameters
    nx = 6
    nu = 4                              # [ux,uy,uz,Γ]
    g = [-9.81, 0, 0]
    k_D = 0.5
    t_N = 5;                        # s, duration of problem
    m = 0.3;                        # kg, mass of quadrotor
    T_min = 1.0;                    # N, minimum thrust
    T_max = 4.0;                    # N, maximum thrust
    theta_max = pi/4;               # rad, maximum tilt angle
    N = 30                          # number of nodes
    
    # initial and final states
    x_initial = [0, 0, 0, 0, 0.5, 0];
    x_final = [0, 10, 0, 0, 0.5, 0];
    
    # obstacle avoidance parameters
    R_obstacle_1 = 1.0              # m, radius of obstacle 1
    p_obstacle_1 = [0, 3, 0.45]     # m, position of obstacle 1
    R_obstacle_2 = 1.0              # m, radius of obstacle 2
    p_obstacle_2 = [0, 7, -0.45]    # m, position of obstacle 2

    
    params = QuadroptorParams(zeros(nu))

    # rhs and jacobian expressions for quadrotor dynamics
    function quadrotor_dfdx(x, u, p, t)
        v = x[4:6]
        v_norm = norm(v)
        dfdx = [zeros(3,3) I(3);
                zeros(3,3)  (-k_D * (v_norm * I(3) + (v * v') / v_norm))]
        return dfdx
    end

    function quadrotor_dfdu(x, u, p, t)
        dfdu = [zeros(3,4); 1/m * I(3) zeros(3,1)];
        return dfdu
    end

    function quadrotor_rhs!(dx, x, p, t)
        dx[1:3] = x[4:6]
        dx[4:6] = -k_D*norm(x[4:6])*x[4:6] + g
        B = quadrotor_dfdu(x[1:6], p.u, p, t)
        dx[:] += B * p.u
        return
    end


    function quadroptor_rhs_aug!(dx_aug, x_aug, p, t)
        A = quadrotor_dfdx(x_aug[1:6], p.u, p, t)
        B = quadrotor_dfdu(x_aug[1:6], p.u, p, t)

        dx_aug[1:3] = x_aug[4:6]
        dx_aug[4:6] = -k_D*norm(x_aug[4:6])*x_aug[4:6] + g
        dx_aug[1:6] += B * p.u

        # derivatives of Phi_A, Phi_B
        dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
    end

    # -------------------- define objective & non-convex constraints -------------------- #
    function objective(x, u)
        return sum(u[4,:])
    end

    nh = 2 * N    # two obstacles, enforced at each node
    function h_noncvx(x,u)
        h = vcat(
            [R_obstacle_1 - norm(x[1:3,k] - p_obstacle_1) for k in 1:N],
            [R_obstacle_2 - norm(x[1:3,k] - p_obstacle_2) for k in 1:N]
        )
        return h
    end

    # -------------------- create problem -------------------- #
    times = LinRange(0.0, t_N, N)

    x_ref = hcat([[el for el in LinRange(x_initial[i], x_final[i], N)] for i in 1:6]...)'
    u_ref = zeros(nu, N-1)
    u_ref[1:3,:] = repeat(-m*g, outer=[1,N-1])
    u_ref[4,:] = norm.(eachcol(u_ref[1:3,:]))
    y_ref = nothing

    # instantiate problem object    
    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        quadrotor_rhs!,
        params,
        objective,
        times,
        x_ref,
        u_ref;
        nh = nh,
        h_noncvx = h_noncvx,
        eom_aug! = quadroptor_rhs_aug!,
        ode_method = Tsit5(),
    )
    set_silent(prob.model)

    # append boundary conditions
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == x_initial)
    @constraint(prob.model, constraint_final_rv,   prob.model[:x][:,end] == x_final)
    @constraint(prob.model, constraint_initial_u, prob.model[:u][1:3,1] == -m * g)
    @constraint(prob.model, constraint_final_u, prob.model[:u][1:3,end] == -m * g)

    # append path constraints
    @constraint(prob.model, constraint_x, prob.model[:x][1,:] == 0)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude_min[k in 1:N-1],
        prob.model[:u][4,k] >= T_min)
    @constraint(prob.model, constraint_control_magnitude_max[k in 1:N-1],
        prob.model[:u][4,k] <= T_max)


    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(nx, N; nh=nh, w0 = 10.0)   # don't forget to pass `nh` to the algorithm as well!

    # solve problem
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        verbosity = verbosity, tol_opt = 1e-6, tol_feas = 1e-6)

    # propagate solution
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @test maximum(abs.(g_dynamics_opt)) <= 1e-6
    @test solution.status == :Optimal
end


test_scvxstar_h_noncvx(verbosity = verbosity)