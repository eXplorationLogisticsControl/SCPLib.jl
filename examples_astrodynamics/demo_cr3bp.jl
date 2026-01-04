"""Example with CR3BP + mass dynamics with free final time"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using ProgressMeter


if !@isdefined(solution)
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))

    # -------------------- setup problem -------------------- #
    # create parameters with `u` entry
    mutable struct ControlParams
        μ::Float64
        c1::Float64
        c2::Float64
        u::Vector
        function ControlParams(μ::Float64, c1::Float64, c2::Float64)
            new(μ, c1, c2, zeros(5))
        end
    end

    μ = 1.215058560962404e-02
    c1 = 0.1
    c2 = 0.1
    DU = 389703     # km
    TU = 382981     # sec
    MU = 500.0      # kg
    VU = DU/TU      # km/s
    params = ControlParams(μ, c1, c2)

    function eom!(drvm, rvm, p, t)
        x, y, z = rvm[1:3]
        vx, vy, vz = rvm[4:6]
        r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
        r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
        drvm[1:3] = rvm[4:6]
        # derivatives of velocities
        drvm[4] =  2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x);
        drvm[5] = -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y;
        drvm[6] = -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z;
        # mass derivative
        drvm[7] = -p.u[4] * p.c2
        # append controls
        drvm[4:6] += p.u[1:3] * p.c1 / rvm[7]
        # time derivative
        drvm[8] = 1.0
        # multiply by time factor
        drvm[1:8] *= p.u[5]
        return
    end

    # boundary conditions
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
    params.u[5] = period_0
    sol_lpo0 = solve(
        ODEProblem(eom!, [rv0; 1.0; 0.0], [0.0, 1.0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )
    params.u[5] = period_f
    sol_lpof = solve(
        ODEProblem(eom!, [rvf; 1.0; 0.0], [0.0, 1.0], params),
        Tsit5(); reltol = 1e-12, abstol = 1e-12
    )

    # -------------------- define objective -------------------- #
    function objective(x, u)
        return -x[7,end] #sum(u[4,:])
    end

    # -------------------- create problem -------------------- #
    N = 100
    nx = 8                              # [x,y,z,vx,vy,vz,mass,time]
    nu = 5                              # [ux,uy,uz,Γ,tf]
    tf = 1.0                            # fixed to unity
    times = LinRange(0.0, tf, N)

    # create reference solution
    x_along_lpo0 = sol_lpo0(LinRange(0.0, 1.0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, 1.0, N))
    x_ref = ones(nx,N)
    alphas = LinRange(0,1,N)
    for (i,alpha) in enumerate(alphas)
        x_ref[1:6,i] = (1-alpha)*x_along_lpo0[1:6,i] + alpha*x_along_lpof[1:6,i]
    end
    tf_guess = 3.0
    x_ref[8,:] = LinRange(0.0, tf_guess, N)
    u_ref = [zeros(nu-1, N-1); tf_guess*ones(1,N-1)]

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
    @constraint(prob.model, constraint_initial_mass, prob.model[:x][7,1] == 1.0)
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][1:6,1] == rv0)
    @constraint(prob.model, constraint_final_rv,   prob.model[:x][1:6,end] == rvf)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4,k] <= 1.0)

    # append constraints on time factor
    tf_span = [2.0, 4.0]
    # @constraint(prob.model, constraint_tf_lb, prob.model[:u][5,1] >= tf_span[1])
    # @constraint(prob.model, constraint_tf_ub, prob.model[:u][5,end] <= tf_span[2])
    # @constraint(prob.model, constraint_tf_uniform[k in 1:N-2],
    #     prob.model[:u][5,k] == prob.model[:u][5,k+1])
    @constraint(prob.model, constraint_tscale_lb[k in 1:N-1], prob.model[:u][5,k] >= tf_span[1])
    @constraint(prob.model, constraint_tscale_ub[k in 1:N-1], prob.model[:u][5,k] <= tf_span[2])
    @constraint(prob.model, constraint_tf_lb, prob.model[:x][8,end] >= tf_span[1])
    @constraint(prob.model, constraint_tf_ub, prob.model[:x][8,end] <= tf_span[2])


    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(nx, N; w0 = 1e2, l1_penalty = false)

    # callback to store solution
    xs_iter = []
    us_iter = []
    function callback(solution)
        push!(xs_iter, deepcopy(solution.x))
        push!(us_iter, deepcopy(solution.u))
    end

    # solve problem
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 1000, callback = callback)
end


# propagate solution
println("Propagating & plotting solution...")
@showprogress for (j_sol, (_x,_u)) in enumerate(zip(xs_iter, us_iter))
    title_str = "Iteration $(@sprintf("%2d", j_sol)): J = $(@sprintf("%1.6f", solution.info[:J0][j_sol])), χ = $(@sprintf("%1.6e", solution.info[:χ][j_sol]))"
    # plot initial guess
    _fig = Figure(size=(400,400))
    _ax3d = Axis3(_fig[1,1]; aspect=:data, protrusions=(50,20,-10,5),
        title=title_str)
    xlims!(_ax3d, 0.905, 1.20)
    ylims!(_ax3d, -0.18, 0.18)
    zlims!(_ax3d, -0.225, 0.125)
    lines!(Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:blue)
    lines!(Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:green)

    _sols, _ = SCPLib.get_trajectory(prob, _x, _u)
    _arc_colors = [
        _u[4,i] > 1e-6 ? :red : :black for i in 1:N-1
    ]
    lw = j_sol == length(xs_iter) ? 1.25 : 1.25
    for (i, _sol) in enumerate(_sols)
        lines!(_ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=_arc_colors[i], linewidth=lw)
    end
    save(joinpath(@__DIR__, "img_demo_cr3bp/cr3bp_iter_$(@sprintf("%2d", 1000000 + j_sol)).png"), _fig; px_per_unit=3)
end
println("Done!")