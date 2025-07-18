"""Quadcoptor example with SCvx*"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


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
N = 30;                         # number of nodes

# initial and final states
x_initial = [0, 0, 0, 0, 0.5, 0];
x_final = [0, 10, 0, 0, 0.5, 0];

# obstacle avoidance parameters
R_obstacle_1 = 1.0              # m, radius of obstacle 1
p_obstacle_1 = [0, 3, 0.45]     # m, position of obstacle 1
R_obstacle_2 = 1.0              # m, radius of obstacle 2
p_obstacle_2 = [0, 7, -0.45]    # m, position of obstacle 2

# ODE parameters
mutable struct QuadroptorParams
    u::Vector
end

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
    dx[1:6] += B * p.u
    return
end

function quadroptor_rhs_aug!(dx_aug, x_aug, p, t)
    quadrotor_rhs!(dx_aug, x_aug, p, t)

    # derivatives of Phi_A, Phi_B
    A = quadrotor_dfdx(x_aug[1:6], p.u, p, t)
    B = quadrotor_dfdu(x_aug[1:6], p.u, p, t)
    dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
    dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
end

# -------------------- define objective & non-convex constraints -------------------- #
function objective(x, u, y)
    return sum(u[4,:])
end

nh = 2 * N    # two obstacles, enforced at each node
function h_noncvx(x,u,y)
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
    u_ref,
    y_ref;
    nh = nh,
    h_noncvx = h_noncvx,
    #eom_aug! = quadroptor_rhs_aug!,   # uncomment to use the user-defined eom_aug!
    ode_method = Tsit5(),
)
set_silent(prob.model)

# append boundary conditions
@constraint(prob.model, constraint_initial_rv, prob.model[:x][:,1] == x_initial)
@constraint(prob.model, constraint_final_rv,   prob.model[:x][:,end] == x_final)
@constraint(prob.model, constraint_initial_u, prob.model[:u][1:3,1] == -m * g)
@constraint(prob.model, constraint_final_u, prob.model[:u][1:3,end] == -m * g)

# append convex path constraints
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
solution = SCPLib.solve!(algo, prob, x_ref, u_ref, y_ref; tol_opt = 1e-6, tol_feas = 1e-6)


# -------------------- analysis of solution -------------------- #
# propagate solution
sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u, solution.y)

# get obstacles x[2] & x[3] values for plotting
coord_obstacle_1 = R_obstacle_1 * [cos.(LinRange(0, 2*pi, 100)) sin.(LinRange(0, 2*pi, 100))]' .+ p_obstacle_1[2:3]
coord_obstacle_2 = R_obstacle_2 * [cos.(LinRange(0, 2*pi, 100)) sin.(LinRange(0, 2*pi, 100))]' .+ p_obstacle_2[2:3]

# plot
fig = Figure(size=(1200,800))
ax2d = Axis(fig[1,1]; xlabel = "East, m", ylabel = "North, m", autolimitaspect=1)
scatter!(ax2d, [x_initial[2]], [x_initial[3]], color=:blue)
scatter!(ax2d, [x_final[2]], [x_final[3]], color=:green)
for (i, _sol) in enumerate(sols_opt)
    lines!(ax2d, Array(_sol)[2,:], Array(_sol)[3,:], color=:black)
end
lines!(ax2d, coord_obstacle_1[1,:], coord_obstacle_1[2,:], color=:red)
lines!(ax2d, coord_obstacle_2[1,:], coord_obstacle_2[2,:], color=:red)

# plot controls
ax_u = Axis(fig[2,1]; xlabel="Time", ylabel="Control")
for i in 1:3
    stairs!(ax_u, prob.times[1:end-1], solution.u[i,:], label="u[$i]", step=:pre, linewidth=1.0)
end
stairs!(ax_u, prob.times[1:end-1], solution.u[4,:], label="||u||", step=:pre, linewidth=2.0, color=:black, linestyle=:dash)
hlines!(ax_u, [-T_max, T_max], color=:red, linestyle=:dot, label="||u|| bounds")
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
scatterlines!(ax_Δ, 1:length(solution.info[:accept]), solution.info[:Δ], color=colors_accept, marker=:circle, markersize=7)

display(fig)
println("Done!")