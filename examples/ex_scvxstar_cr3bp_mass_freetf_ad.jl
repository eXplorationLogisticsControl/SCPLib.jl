"""Example with CR3BP + mass dynamics with free final time"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

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
    # multiply by time factor
    drvm[1:7] *= p.u[5]
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
    ODEProblem(eom!, [rv0; 1.0], [0.0, 1.0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
params.u[5] = period_f
sol_lpof = solve(
    ODEProblem(eom!, [rvf; 1.0], [0.0, 1.0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)

# -------------------- define objective -------------------- #
function objective(x, u, y)
    return -x[7,end] #sum(u[4,:])
end

# -------------------- create problem -------------------- #
N = 50
nx = 7
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
u_ref = [zeros(nu-1, N-1); tf*ones(1,N-1)];
y_ref = nothing

# plot initial guess
fig = Figure(size=(1600,800))
ax3d = Axis3(fig[1,1]; aspect=:data)
lines!(Array(sol_lpo0)[1,:], Array(sol_lpo0)[2,:], Array(sol_lpo0)[3,:], color=:blue)
lines!(Array(sol_lpof)[1,:], Array(sol_lpof)[2,:], Array(sol_lpof)[3,:], color=:green)

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
@constraint(prob.model, constraint_tf_lb, prob.model[:u][5,1] >= tf_span[1])
@constraint(prob.model, constraint_tf_ub, prob.model[:u][5,end] <= tf_span[2])
@constraint(prob.model, constraint_tf_uniform[k in 1:N-2],
    prob.model[:u][5,k] == prob.model[:u][5,k+1])


# -------------------- instantiate algorithm -------------------- #
algo = SCPLib.SCvxStar(nx, N; w0 = 1e4)

# solve problem
solution = SCPLib.solve!(algo, prob, x_ref, u_ref, y_ref; maxiter = 100)

# propagate solution
sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u, solution.y)
arc_colors = [
    solution.u[4,i] > 1e-6 ? :red : :black for i in 1:N-1
]
for (i, _sol) in enumerate(sols_opt)
    lines!(ax3d, Array(_sol)[1,:], Array(_sol)[2,:], Array(_sol)[3,:], color=arc_colors[i])
end

# plot controls
ax_u = Axis(fig[2,1]; xlabel="Time", ylabel="Control")
for i in 1:3
    stairs!(ax_u, prob.times[1:end-1] .* solution.u[5,:], solution.u[i,:], label="u[$i]", step=:pre, linewidth=1.0)
end
stairs!(ax_u, prob.times[1:end-1] .* solution.u[5,:], solution.u[4,:], label="||u||", step=:pre, linewidth=2.0, color=:black, linestyle=:dash)
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

ax_m = Axis(fig[1,4]; xlabel="Time", ylabel="mass")
for (i, _sol) in enumerate(sols_opt)
    lines!(ax_m, _sol.t * solution.u[5,i], Array(_sol)[7,:], color=arc_colors[i])
end

display(fig)
println("Done!")