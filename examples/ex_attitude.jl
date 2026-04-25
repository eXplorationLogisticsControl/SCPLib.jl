"""Example for attitude control problem"""

using LinearAlgebra
using Clarabel
using GLMakie
using JuMP
using OrdinaryDiffEq
using ForwardDiff

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

function quat2rotm(q::Vector{T}) where T<:Real
    qw, qx, qy, qz = q
    R = [
        1 - 2*(qy^2 + qz^2)    2*(qx*qy - qz*qw)      2*(qx*qz + qy*qw);
        2*(qx*qy + qz*qw)      1 - 2*(qx^2 + qz^2)    2*(qy*qz - qx*qw);
        2*(qx*qz - qy*qw)      2*(qy*qz + qx*qw)      1 - 2*(qx^2 + qy^2)
    ]
    return R
end

# Parameters
nx = 7
nu = 4
mutable struct Params
    u::Vector{Float64}
    I::Matrix{Float64}
    invI::Matrix{Float64}

    function Params(u::Vector{Float64}, I::Matrix{Float64})
        new(u, I, inv(I))
    end
end

inertia_matrix = diagm([1.0, 2.0, 3.0])
params = Params(zeros(nu), inertia_matrix)

# derivative function
p_dual = deepcopy(params)
f_u2dx! = function (dx,x,u,t)
    p_dual.u = u             # replace control with dual vector
    eom!(view(dx,1:nx),x[1:nx],p_dual,t)
end

# Dynamics
#   quaterinon convention: q = [q_scalar, q_vector]
function eom!(dx, x, params, t)
    q = x[1:4]
    ω = x[5:7]
    Ω = [-q[2] -q[3] -q[4];
         q[1] -q[4]  q[3];
         q[4]  q[1] -q[2];
        -q[3]  q[2]  q[1]]
    dx[1:4] = 0.5 * Ω * ω
    τ = params.u[1:3]
    Iω = params.I * ω
    dx[5:7] = params.invI * (params.u[1:3] - cross(ω, Iω))
    return
end

function eom_aug!(dx_aug, x_aug, params, t)
    eom!(view(dx_aug,1:7), x_aug[1:7], params, t)
    A = ForwardDiff.jacobian((y,x) -> eom!(y,x,params,t), zeros(7), x_aug[1:7])
    B = zeros(nx,nu)
    B[5:7,1:3] = params.invI

    dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx)), nx*nx)
    dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nx,nu)) + B), nx*nu)
    return
end

# Objective
function objective(x, u)
    return sum(u[4,:])
end

# # Constraints
# ω_max = 1.0
# X_init = ([0,0,0,1, 0,0,0], [0,0,0,1, 0,0,0])  # exact bounds
# X_term = ([1,0,0,0, 0,0,0], [1,0,0,0, 0,0,0])
# X_path(t) = ([-Inf,-Inf,-Inf,-Inf, -ω_max,-ω_max,-ω_max], [Inf,Inf,Inf,Inf, ω_max,ω_max,ω_max])

function eul2quat(eul::Vector{Float64})
    q = [cos(eul[3]/2), sin(eul[3]/2), 0.0, 0.0]
    return q
end

# Initial guesses
N = 60
times = collect(LinRange(0.0, 10.0, N))
x_ref = zeros(nx, N)
q0 = [0.0, 0.0, 0.4, 1.0]
qf = [1.0, 0.0, 0.0, 0.0]

q0 /= norm(q0)
qf /= norm(qf)

for k in 1:N
    s = (k-1)/(N-1)
    q_ref = (1-s)*q0 + s*qf
    q_ref /= norm(q_ref)
    x_ref[1:4, k] = q_ref
    x_ref[5:7, k] = [0.0, 0.0, 0.0]
end
u_ref = zeros(4, N-1)

# Problem definition
prob = SCPLib.ContinuousProblem(
    Clarabel.Optimizer,
    eom!,
    params,
    objective,
    times,
    x_ref,
    u_ref;
    eom_aug! = eom_aug!,
)
set_silent(prob.model)

# append boundary conditions
@constraint(prob.model, constraint_initial_q, prob.model[:x][1:4,1] == q0)
@constraint(prob.model, constraint_final_q,   prob.model[:x][1:4,end] == qf)
@constraint(prob.model, constraint_initial_ω, prob.model[:x][5:7,1] == [0.0, 0.0, 0.0])
@constraint(prob.model, constraint_final_ω,   prob.model[:x][5:7,end] == [0.0, 0.0, 0.0])

# append constraints on control magnitude
τ_max = 0.5
@constraint(prob.model, constraint_associate_control[k in 1:N-1],
    [prob.model[:u][4,k], prob.model[:u][1:3,k]...] in SecondOrderCone())
@constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
    prob.model[:u][4,k] <= τ_max)

# -------------------- instantiate algorithm -------------------- #
algo = SCPLib.SCvxStar(nx, N; w0 = 1e2)

# solve problem
solution = SCPLib.solve!(algo, prob, x_ref, u_ref; maxiter = 100)

# propagate solution
sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)

# plot state history & trajectory
state_labels = ["q1", "q2", "q3", "q4", "ω1", "ω2", "ω3"]
fig = Figure(size=(1200,800))
for i in 1:7
    irow = div(i-1, 4) + 1
    icol = mod(i-1, 4) + 1
    _ax = Axis(fig[irow,icol]; xlabel="Time", ylabel=state_labels[i])
    for (isol, _sol) in enumerate(sols_opt)
        lines!(_ax, _sol.t, Array(_sol)[i,:], color=:black)
    end
end

axu = Axis(fig[2,4]; xlabel="Time", ylabel="Control")
for i in 1:3
    stairs!(axu, times[1:end-1], solution.u[i,:], label="u[$i]", step=:pre, linewidth=1.0)
end
stairs!(axu, times[1:end-1], solution.u[4,:], label="||u||", step=:pre, linewidth=2.0, color=:black, linestyle=:dash)
axislegend(axu, position=:cc)

display(fig)

# plot attitude in 3D
fig_attitude = Figure(size=(700,500))
ax3d = Axis3(fig_attitude[1,1];
    aspect=:data, xlabel="Time")
# hidedecorations!()
xlims!(ax3d, -1, 11)
ylims!(ax3d, -1, 1)
zlims!(ax3d, -1, 1)
origin_xs = times #collect(LinRange(0.0, 10.0, N))
scale_axes = 0.7
for k in 1:4:N
    dcm_k = scale_axes * quat2rotm(solution.x[1:4,k])
    i1 = dcm_k[1,:]
    i2 = dcm_k[2,:]
    i3 = dcm_k[3,:]
    scatter!(ax3d, origin_xs[k], 0.0, 0.0, color=:black)
    # arrows!([origin_xs[k], origin_xs[k]+i1[1]], [0.0, i1[2]], [0.0, i1[3]], color=:red)
    # lines!(ax3d, [origin_xs[k], origin_xs[k]+i2[1]], [0.0, i2[2]], [0.0, i2[3]], color=:blue)
    # lines!(ax3d, [origin_xs[k], origin_xs[k]+i3[1]], [0.0, i3[2]], [0.0, i3[3]], color=:green)
    arrows!(ax3d, [origin_xs[k]], [0.0], [0.0], [i1[1]], [i1[2]], [i1[3]], color=:red)
    arrows!(ax3d, [origin_xs[k]], [0.0], [0.0], [i2[1]], [i2[2]], [i2[3]], color=:blue)
    arrows!(ax3d, [origin_xs[k]], [0.0], [0.0], [i3[1]], [i3[2]], [i3[3]], color=:green)
end

display(fig_attitude)

println("Done!")
