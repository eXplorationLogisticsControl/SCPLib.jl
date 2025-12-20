"""Two-body transfer problem"""

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

MU_SUN = 132712000000.0
G0 = 9.81
DU = 149.6e6
VU = sqrt(MU_SUN / DU)          # velocity scale, m/s
TU = DU / VU                    # time scale, s
MASS = 2000.0                   # kg

THRUST = 0.4                    # Newtons
ISP = 3000.0                    # seconds

μ  = MU_SUN / (VU^2 * DU)
c1 = THRUST/1e3 / (MASS*DU/TU^2)               # canonical max thrust
c2 = THRUST/1e3 / (ISP*G0/1e3) / (MASS/TU)     # canonical mass flow rate
params = ControlParams(μ, c1, c2)

function eom!(drvm, rvm, p, t)
    drvm[1:3] =  rvm[4:6]
    drvm[4:6] = -p.μ / norm(rvm[1:3])^3 * rvm[1:3] + p.u[1:3] * p.c1 / rvm[7]
    drvm[7]   = -p.u[4] * p.c2
    drvm[8]   = 1.0                 # time

    # multiply by time factor
    drvm[1:8] *= p.u[5]
    return
end

rv0 = [1.0, 0.0, 0.0, 0.0, sqrt(μ/1.0), 0.0]
rvf = [1.5, 0.0, 0.0, 0.0, sqrt(μ/1.5), 0.1]
period_0 = 2π * sqrt(1.0^3/μ)
period_f = 2π * sqrt(1.5^3/μ)

# initial & final orbits
params.u[5] = period_0
sol_orbit0 = solve(
    ODEProblem(eom!, [rv0; 1.0; 0.0], [0.0, 1.0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
params.u[5] = period_f
sol_orbitf = solve(
    ODEProblem(eom!, [rvf; 1.0; 0.0], [0.0, 1.0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)

# -------------------- define objective -------------------- #
function objective(x, u, y) return -x[7,end] end

# -------------------- create problem -------------------- #
N = 100
nx = 8                              # [x,y,z,vx,vy,vz,mass,time]
nu = 5                              # [ux,uy,uz,Γ,tf]
tf = 1.0                            # fixed to unity
times = LinRange(0.0, tf, N)

# create reference solution
tf_guess = 2π
x_along_orbit0 = sol_orbit0(LinRange(0.0, 1.0, N))
x_along_orbitf = sol_orbitf(LinRange(0.0, 1.0, N))
x_ref = ones(nx,N)
alphas = LinRange(0,1,N)
for (i,alpha) in enumerate(alphas)
    x_ref[1:6,i] = (1-alpha)*x_along_orbit0[1:6,i] + alpha*x_along_orbitf[1:6,i]
end
x_ref[1:6,end] = rvf[1:6]   # to avoid initial infeasibility
x_ref[8,:] = LinRange(0.0, tf_guess, N)
u_ref = [zeros(nu-1, N-1); tf_guess*ones(1,N-1)]

# plot initial guess
fig = Figure(size=(1600,800))
ax3d = Axis3(fig[1,1]; aspect=:data)
lines!(Array(sol_orbit0)[1,:], Array(sol_orbit0)[2,:], Array(sol_orbit0)[3,:], color=:blue)
lines!(Array(sol_orbitf)[1,:], Array(sol_orbitf)[2,:], Array(sol_orbitf)[3,:], color=:green)

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
tf_span = [1.5π, 3π]
# @constraint(prob.model, constraint_tf_lb, prob.model[:u][5,1] >= tf_span[1])
# @constraint(prob.model, constraint_tf_ub, prob.model[:u][5,end] <= tf_span[2])
# @constraint(prob.model, constraint_tf_uniform[k in 1:N-2],
#     prob.model[:u][5,k] == prob.model[:u][5,k+1])
@constraint(prob.model, constraint_tscale_lb[k in 1:N-1], prob.model[:u][5,k] >= tf_span[1])
@constraint(prob.model, constraint_tscale_ub[k in 1:N-1], prob.model[:u][5,k] <= tf_span[2])
@constraint(prob.model, constraint_tf_lb, prob.model[:x][8,end] >= tf_span[1])
@constraint(prob.model, constraint_tf_ub, prob.model[:x][8,end] <= tf_span[2])


# -------------------- instantiate algorithm -------------------- #
algo = SCPLib.SCvxStar(nx, N; w0 = 1e2, w_max=1e20)

# solve problem
solution = SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 100)

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
times_u = []
umags = []
udirs = []
for (i, _sol) in enumerate(sols_opt)
    u_zoh = solution.u[4,i] * ones(length(_sol.t))
    append!(times_u, Array(_sol)[8,:])
    append!(umags, u_zoh)
    push!(udirs, [solution.u[1,i] * ones(1,length(_sol.t)); solution.u[2,i] * ones(1,length(_sol.t)); solution.u[3,i] * ones(1,length(_sol.t))])
end
udirs = hcat(udirs...)
for i in 1:3
    stairs!(ax_u, times_u, udirs[i,:], label="u[$i]", step=:pre, linewidth=1.0)
end
stairs!(ax_u, times_u, umags, label="||u||", step=:pre, linewidth=0.5, color=:black)
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
    lines!(ax_m, Array(_sol)[8,:], Array(_sol)[7,:], color=arc_colors[i])
end

display(fig)
println("Done!")