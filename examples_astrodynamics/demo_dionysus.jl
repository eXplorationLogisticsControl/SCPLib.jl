"""Sample SCP problem with MEE"""

using Clarabel
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using ProgressMeter

using AstrodynamicsCore

if !@isdefined(solution__)
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))

    mutable struct ControlParams
        μ::Float64
        c1::Float64
        c2::Float64
        u::Vector
        function ControlParams(μ::Float64, c1::Float64, c2::Float64)
            new(μ, c1, c2, zeros(4))
        end
    end

    MU_SUN = 132712440018.0
    G0 = 9.8065
    DU = 149.6e6
    VU = sqrt(MU_SUN / DU)          # velocity scale, m/s
    TU = DU / VU                    # time scale, s
    MASS = 4000.0                   # kg

    THRUST = 0.32                    # Newtons
    ISP = 3000.0                    # seconds

    μ  = MU_SUN / (VU^2 * DU)
    c1 = THRUST/1e3 / (MASS*DU/TU^2)               # canonical max thrust
    c2 = THRUST/1e3 / (ISP*G0/1e3) / (MASS/TU)     # canonical mass flow rate
    params = ControlParams(μ, c1, c2)

    function eom_mee!(drvm, rvm, params, t)
        p,f,g,h,k,L,mass = rvm[1:7]   # unpack state
        cosL = cos(L)
        sinL = sin(L)
        s2 = 1 + h^2 + k^2
        w = 1 + f*cosL + g*sinL
        hsinL_kcosL = h*sinL - k*cosL
        B_mee = sqrt(p/params.μ) * [
            0    2p/w                0;
            sinL ((1+w)*cosL + f)/w -g/w*hsinL_kcosL;
            -cosL ((1+w)*sinL + g)/w  f/w*hsinL_kcosL;
            0    0                   1/w*s2/2*cosL;
            0    0                   1/w*s2/2*sinL;
            0    0                   1/w*hsinL_kcosL;
        ]
        D = [0.0, 0.0, 0.0, 0.0, 0.0, sqrt(params.μ/p^3) * (1 + f*cosL + g*sinL)^2]
        drvm[1:6] = B_mee * (params.c1 / mass) * params.u[1:3] + D
        drvm[7] = -params.u[4] * params.c2
        return
    end

    # -------------------- boundary conditions -------------------- #
    tof = 3534 * 86400 / TU
    N_rev = 5

    R0 = [-3637871.081; 147099798.784; -2261.44] / DU
    V0 = [-30.265097; -0.8486854; 0.0000505] / VU
    RV0 = [R0; V0]
    orbit_init = AstrodynamicsCore.Planet(params.μ, 0.0, RV0, "initial")
    mee0 = AstrodynamicsCore.rv2mee([R0; V0], params.μ)

    M0 = deg2rad(114.4232)
    TA0 = AstrodynamicsCore.ma2ta(M0, 0.542)
    kepf = [2.2, 0.542, deg2rad(13.6), deg2rad(82.2), deg2rad(204.2), TA0]
    RVf0 = AstrodynamicsCore.kep2rv(kepf, params.μ)
    orbit_final = AstrodynamicsCore.Planet(params.μ, 0.0, RVf0, "final")
    RVf = AstrodynamicsCore.eph(orbit_final, tof + (56284 - 53400)  *86400/TU)
    meef = AstrodynamicsCore.rv2mee(RVf, params.μ)
    meef[6] += 2π * N_rev   # append revolutions

    x0_ref = [mee0; 1.0]
    xf_ref = [meef; 0.4]

    # get initial and final orbits for plotting
    initial_orbit_rvs = hcat([AstrodynamicsCore.eph(orbit_init, t) for t in LinRange(0.0, orbit_init.period, 300)]...)
    final_orbit_rvs = hcat([AstrodynamicsCore.eph(orbit_final, t) for t in LinRange(0.0, orbit_final.period, 300)]...)

    # -------------------- define objective -------------------- #
    function objective(x, u)
        return -x[7,end]
    end

    ng = 6
    function g_noncvx(x, u)
        g = AstrodynamicsCore.mee2rv(x[1:6,end], params.μ) - RVf
        return g
    end


    # -------------------- create problem -------------------- #
    N = 500
    nx = 7                              # [p,f,g,h,k,L,mass]
    nu = 4                              # [ux,uy,uz,Γ]
    times = LinRange(0.0, tof, N)

    # create reference solution
    x_ref = zeros(nx, N)
    x_ref[1:6,:] = hcat(LinRange.(mee0, meef, N)...)'
    x_ref[7,:] = LinRange(x0_ref[7], xf_ref[7], N)
    u_ref = zeros(nu, N-1)

    # instantiate problem object    
    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom_mee!,
        params,
        objective,
        times,
        x_ref,
        u_ref;
        ng = ng,
        g_noncvx = g_noncvx,
        ode_method = Vern7(),
    )
    set_silent(prob.model)

    # append boundary conditions
    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == x0_ref)
    # @constraint(prob.model, constraint_final_rv,   prob.model[:x][1:6,end] == meef)  # enforced in g_noncvx

    # minimum on mass for numerical stability
    @constraint(prob.model, constraint_monotonic_ma[k in 1:N-1], prob.model[:x][6,k] <= prob.model[:x][6,k+1])
    @constraint(prob.model, constraint_mass_lb[k in 1:N], prob.model[:x][7,k] >= 0.1)
    # @constraint(prob.model, constraint_mass_ub[k in 1:N], prob.model[:x][7,k] <= 1.0)
    @constraint(prob.model, constraint_p_lb[k in 1:N], prob.model[:x][1,k] >= 0.8)

    # append constraints on control magnitude
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4,k] <= 1.0)


    sols_ig, _ = SCPLib.get_trajectory(prob, x_ref, u_ref)

    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(nx, N; ng=ng, w0 = 1e0, w_max=1e20)  # known to work: w0 = 1e0 with N = 500

    # callback to store solution
    xs_iter = []
    us_iter = []
    function callback(solution)
        push!(xs_iter, deepcopy(solution.x))
        push!(us_iter, deepcopy(solution.u))
    end

    # solve problem
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 500, callback = callback)
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @show -solution.info[:J0][end] * MASS
end

# propagate solution
println("Propagating & plotting solution...")
@showprogress for (j_sol, (_x,_u)) in enumerate(zip(xs_iter, us_iter))
    title_str = "Iteration $(@sprintf("%3d", j_sol)): J = $(@sprintf("%1.6e", solution.info[:J0][j_sol])), χ = $(@sprintf("%1.6e", solution.info[:χ][j_sol]))"
    # plot initial guess
    _fig = Figure(size=(800,400))
    _ax3d = Axis3(_fig[1,1]; aspect=:data, protrusions=(50,60,0,5),
        title=title_str)
    xlims!(_ax3d, -2.4, 2.1)
    ylims!(_ax3d, -1.75, 4.0)
    zlims!(_ax3d, -0.8, 0.8)

    scatter!(_ax3d, RV0[1], RV0[2], RV0[3], color=:limegreen, markersize=10)
    scatter!(_ax3d, RVf[1], RVf[2], RVf[3], color=:blue, markersize=10)
    lines!(_ax3d, initial_orbit_rvs[1,:], initial_orbit_rvs[2,:], initial_orbit_rvs[3,:], color=:limegreen, linewidth=0.8)
    lines!(_ax3d, final_orbit_rvs[1,:], final_orbit_rvs[2,:], final_orbit_rvs[3,:], color=:blue, linewidth=0.8)

    _sols, _ = SCPLib.get_trajectory(prob, _x, _u)
    _arc_colors = [
        _u[4,i] > 1e-6 ? :red : :black for i in 1:N-1
    ]
    lw = j_sol == length(xs_iter) ? 1.25 : 1.25
    for (isol, _sol) in enumerate(_sols)
        rvs = hcat([AstrodynamicsCore.mee2rv(Array(_sol)[1:6,i], params.μ) for i in 1:length(_sol.t)]...)
        lines!(_ax3d, rvs[1,:], rvs[2,:], rvs[3,:], color=_u[4,isol] > ucolor_tol ? :red : :black, linewidth=1.0)
    end
    save(joinpath(@__DIR__, "img_demo_dionysus/dionysus_iter_$(@sprintf("%2d", 1000000 + j_sol)).png"), _fig; px_per_unit=3)
end
println("Done!")