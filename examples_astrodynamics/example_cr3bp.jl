"""Advanced CR3BP problem example"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))
include(joinpath(@__DIR__, "../../ShootingStar.jl/src/ShootingStar.jl"))


# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU/TU      # km/s
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

# boundary conditions
rv0 = [1.01882617355496, 0.0, -0.179797844569828, 0.0, -0.096189089845127, 0.0]
period_0 = 6.4 * 86400 / TU

rvf = [0.823383959653906, 0.0, 0.010388134109586, 0.0, 0.128105259453086, 0.0]
period_f = 12.0 * 86400 /TU
tf = 57.4 * 86400 / TU

# rv0 = [1.0809931218390707E+00,
#     0.0000000000000000E+00,
#     -2.0235953267405354E-01,
#     1.0157158264396639E-14,
#     -1.9895001215078018E-01,
#     7.2218178975912707E-15]
# period_0 = 2.3538670417546639E+00

# rvf = [1.1648780946517576,
#     0.0,
#     -1.1145303634437023E-1,
#     0.0,
#     -2.0191923237095796E-1,
#     0.0]
# period_f = 3.3031221822879884
# tf = 2.6

# initial & final LPO
sol_lpo0 = solve(
    ODEProblem(eom!, rv0, [0.0, period_0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
sol_lpof = solve(
    ODEProblem(eom!, rvf, [0.0, period_f], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)

# -------------------- define objective -------------------- #
function objective(x, u)
    return sum(u[4,:])
end

# -------------------- create problem -------------------- #
nx = 6
nu = 4                              # [ux,uy,uz,Γ]
# thrust = 0.35    # N
# umax = thrust/MU/1e3 / (VU/TU)
umax = 0.273 * 1e-6 / (DU/TU^2)

if !@isdefined(x_ref)
    N = 100
    Nseg = N - 1
    times = [el for el in LinRange(0.0, tf, Nseg+1)]
    
    params_ode = [μ,]
    nodes = ShootingStar.initialguess_gradual_transit(rv0, rvf, times, 
        ShootingStar.rhs_cr3bp_sv!, params_ode)
    
    # create problem
    prob = ShootingStar.TwoStageShootingProblem(
        rv0,
        rvf,
        times,
        nodes,
        ShootingStar.rhs_cr3bp_svstm!,
        params_ode,
    )

    # solve outerloop
    maxiter = 10
    status, sols, residuals = ShootingStar.solve_outerloop!(
        prob,
        maxiter,
        1e-6;
        verbosity = 1,
        verbosity_inner = 0,
        eps_inner = 1e-8,
    )

    # nodes = ShootingStar.initialguess_gradual_transit(
    #     rv0, rvf, times, 
    #     ShootingStar.rhs_cr3bp_sv!, params_ode)
    x_ref = nodes
    u_ref = hcat(
        (nodes[4,2:end] - nodes[4,1:end-1]) ./ (times[2:end] - times[1:end-1]),
        (nodes[5,2:end] - nodes[5,1:end-1]) ./ (times[2:end] - times[1:end-1]),
        (nodes[6,2:end] - nodes[6,1:end-1]) ./ (times[2:end] - times[1:end-1]),
    )'
    u_ref = [u_ref; umax*ones(1,Nseg)]
    x_ref[:,1] = rv0
    x_ref[:,end] = rvf
end

# plot initial guess
fig = Figure(size=(1200,800))
ax3d = Axis3(fig[1,1]; aspect=:data)
lines!(Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:blue)
lines!(Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:green)

# instantiate problem object    
_x_ref, _u_ref = deepcopy(x_ref), deepcopy(u_ref)
prob = SCPLib.ContinuousProblem(
    Clarabel.Optimizer,
    eom!,
    params,
    objective,
    times,
    _x_ref,
    _u_ref;
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

# initial guess
sol_ig, g_dynamics_ig = SCPLib.get_trajectory(prob, _x_ref, _u_ref)
for _sol in sol_ig
    lines!(ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=:grey)
end

# -------------------- instantiate algorithm -------------------- #
algo = SCPLib.SCvxStar(nx, N; w0 = 1e0)

# solve problem
solution = SCPLib.solve!(algo, prob, _x_ref, _u_ref; maxiter = 500)

# propagate solution
sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
arc_colors = [
    solution.u[4,i] > 1e-6 ? :red : :black for i in 1:N-1
]
for (i, _sol) in enumerate(sols_opt)
    lines!(ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=arc_colors[i])
end

# plot controls
ax_u = Axis(fig[2,1]; xlabel="Time, day", ylabel="Control, mm/s^2")
for i in 1:3
    stairs!(ax_u, prob.times[1:end-1]*TU/86400, solution.u[i,:] * (DU/TU^2)/1e-6, label="u[$i]", step=:pre, linewidth=1.0)
end
stairs!(ax_u, prob.times[1:end-1]*TU/86400, solution.u[4,:] * (DU/TU^2)/1e-6, label="||u||", step=:pre, linewidth=2.0, color=:black, linestyle=:dash)
hlines!(ax_u, [ umax * (DU/TU^2)/1e-6], color=:grey, linestyle=:dot)
hlines!(ax_u, [-umax * (DU/TU^2)/1e-6], color=:grey, linestyle=:dot)
axislegend(ax_u, position=:cc)

# plot iterate information
colors_accept = [solution.info[:accept][i] ? :green : :red for i in 1:length(solution.info[:accept])] 
ax_χ = Axis(fig[1,2]; xlabel="Iteration", ylabel="χ", yscale=log10)
scatterlines!(ax_χ, 1:length(solution.info[:accept]), solution.info[:χ], color=colors_accept, marker=:circle, markersize=7)

ax_w = Axis(fig[2,2]; xlabel="Iteration", ylabel="w", yscale=log10)
scatterlines!(ax_w, 1:length(solution.info[:accept]), solution.info[:w], color=colors_accept, marker=:circle, markersize=7)

ax_J = Axis(fig[1,3]; xlabel="Iteration", ylabel="ΔJ", yscale=log10)
scatterlines!(ax_J, 1:length(solution.info[:accept]), abs.(solution.info[:ΔJ]), color=colors_accept, marker=:circle, markersize=7)

ax_Δ = Axis(fig[2,3]; xlabel="Iteration", ylabel="trust region radius", yscale=log10)
scatterlines!(ax_Δ, 1:length(solution.info[:accept]), [minimum(val) for val in solution.info[:Δ]], color=colors_accept, marker=:circle, markersize=7)

display(fig)
println("Done!")