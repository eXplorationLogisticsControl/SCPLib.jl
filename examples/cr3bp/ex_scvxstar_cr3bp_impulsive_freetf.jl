"""Impulsive, variable-time CR3BP transfer between two Lyapunov orbits with SCvx*.

The transfer is flown with impulsive Δv only; the flight time is free.
The independent variable is `τ ∈ [0, 1]` on `N` nodes, and the state is
`[rx, ry, rz, vx, vy, vz, t]`, where `t` is the physical time.
Each segment coasts as `dx/dτ = s_k [f_cr3bp(r, v); 1]`, with the time dilation `s_k`
as the only continuous control (nodes `1:N-1`). The impulsive control is
`[Δvx, Δvy, Δvz, Γ]` with `‖Δv‖ ≤ Γ ≤ Δv_max`, applied at every node including
the arrival node `N`. The objective is the total `ΣΓ`.

This is solved as a `MixedProblem`: an `ImpulsiveProblem` cannot be used, as its
linearization does not account for the control entering the coast (the dilation).
"""

using CairoMakie
using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../../src/SCPLib.jl"))


# -------------------- setup problem -------------------- #
struct ControlParams
    μ::Float64
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU/TU      # km/s
params = ControlParams(μ)

"""CR3BP vector field, with the physical time rate appended"""
function cr3bp_field!(g, x, μ)
    xx, yy, zz = x[1], x[2], x[3]
    vx, vy, vz = x[4], x[5], x[6]
    r1 = sqrt((xx + μ)^2 + yy^2 + zz^2)
    r2 = sqrt((xx - 1 + μ)^2 + yy^2 + zz^2)
    g[1] = vx
    g[2] = vy
    g[3] = vz
    g[4] =  2*vy + xx - ((1 - μ)/r1^3)*(μ + xx) + (μ/r2^3)*(1 - μ - xx)
    g[5] = -2*vx + yy - ((1 - μ)/r1^3)*yy - (μ/r2^3)*yy
    g[6] = -((1 - μ)/r1^3)*zz - (μ/r2^3)*zz
    g[7] = 1.0
    return g
end

# coast: `u` is the continuous control slice only, i.e. the time dilation
function eom!(dx, x, pu, t)
    (; params, u) = pu
    cr3bp_field!(dx, x, params.μ)
    dx[1:7] .*= u[1]
    return
end

# impulse: velocity jump from [Δvx, Δvy, Δvz]; Γ and the time are unchanged
function dfdu(x, u_i, t)
    return [zeros(3, 4); I(3) zeros(3, 1); zeros(1, 4)]
end

# augmented coast: state, Φ_A, and Φ_B of the continuous control (the dilation) only
function eom_aug!(dx, xaug, pu, t)
    (; params, u) = pu
    s = u[1]
    x = @view(xaug[1:7])
    g = zeros(eltype(xaug), 7)
    cr3bp_field!(g, x, params.μ)
    dx[1:7] = s .* g

    xx, yy, zz = x[1], x[2], x[3]
    r1vec = [xx + params.μ, yy, zz]
    r2vec = [xx - 1 + params.μ, yy, zz]
    G1 = (1 - params.μ) / norm(r1vec)^5 * (3*r1vec*r1vec' - norm(r1vec)^2*I(3))
    G2 = params.μ / norm(r2vec)^5 * (3*r2vec*r2vec' - norm(r2vec)^2*I(3))
    Omega = [0.0 2.0 0.0; -2.0 0.0 0.0; 0.0 0.0 0.0]
    A6 = [zeros(3, 3) I(3); G1 + G2 + diagm([1.0, 1.0, 0.0]) Omega]
    Ag = zeros(eltype(xaug), 7, 7)
    Ag[1:6, 1:6] = A6
    A = s * Ag
    Φ = reshape(@view(xaug[8:56]), 7, 7)
    ΦB = @view(xaug[57:63])
    dx[8:56] = reshape(A * Φ, 49)
    dx[57:63] = A * ΦB + g
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
coast! = function (dx, x, p, t)
    cr3bp_field!(dx, x, μ)
    dx[7] = 0.0
end
sol_lpo0 = solve(ODEProblem(coast!, [rv0; 0.0], (0.0, period_0), nothing), Tsit5(); reltol = 1e-12, abstol = 1e-12)
sol_lpof = solve(ODEProblem(coast!, [rvf; 0.0], (0.0, period_f), nothing), Tsit5(); reltol = 1e-12, abstol = 1e-12)


# -------------------- define objective -------------------- #
# total Δv magnitude; the last control row is Γ (the arrival-node dilation slot u[1,N] is NaN and unused)
function objective(x, u)
    return sum(u[size(u, 1), k] for k in 1:size(u, 2))
end


# -------------------- create problem -------------------- #
N = 21
nx = 7
nuc = 1                             # [s]
nui = 4                             # [Δvx, Δvy, Δvz, Γ]
nu = nuc + nui
times = LinRange(0.0, 1.0, N)
tf_guess = 3.0
tf_span = [2.0, 4.0]

thrust = 0.35    # N, used to scale the maximum impulse magnitude
Δv_max = thrust/MU/1e3 / (VU/TU)

# create reference solution: blend of the two LPOs, uniform dilation, no impulses
x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
x_ref = zeros(nx, N)
for (i, alpha) in enumerate(LinRange(0, 1, N))
    x_ref[1:6, i] = (1 - alpha)*x_along_lpo0[1:6, i] + alpha*x_along_lpof[1:6, i]
end
x_ref[7, :] = LinRange(0.0, tf_guess, N)

u_ref = zeros(nu, N)
u_ref[1, 1:N-1] .= tf_guess
u_ref[1, N] = NaN                   # no dilation at the arrival node

# instantiate problem object
prob = SCPLib.MixedProblem(
    Clarabel.Optimizer,
    eom!,
    params,
    objective,
    times,
    x_ref,
    u_ref;
    nuc = nuc,
    nui = nui,
    dfdu = dfdu,
    eom_aug! = eom_aug!,
    ode_method = Vern7(),
)
set_silent(prob.model)

# boundary conditions: the epoch is pinned, so the final clock is the time of flight;
# the arrival impulse is applied at node N
@constraint(prob.model, constraint_initial_rv, prob.model[:x][1:6, 1] .== rv0)
@constraint(prob.model, constraint_initial_time, prob.model[:x][7, 1] == 0.0)
@constraint(prob.model, constraint_final_r, prob.model[:x][1:3, end] .== rvf[1:3])
@constraint(prob.model, constraint_final_v, prob.model[:x][4:6, end] + prob.model[:ui][1:3, end] .== rvf[4:6])

# impulse magnitude
@constraint(prob.model, constraint_associate_control[k in 1:N],
    [prob.model[:ui][4, k], prob.model[:ui][1:3, k]...] in SecondOrderCone())
@constraint(prob.model, constraint_control_magnitude[k in 1:N],
    prob.model[:ui][4, k] <= Δv_max)

# time of flight and per-segment dilation (not forced to be uniform)
@constraint(prob.model, constraint_dilation_lb[k in 1:N-1], prob.model[:uc][1, k] >= tf_span[1])
@constraint(prob.model, constraint_dilation_ub[k in 1:N-1], prob.model[:uc][1, k] <= tf_span[2])
@constraint(prob.model, constraint_tf_lb, prob.model[:x][7, end] >= tf_span[1])
@constraint(prob.model, constraint_tf_ub, prob.model[:x][7, end] <= tf_span[2])


# -------------------- instantiate algorithm -------------------- #
Δ0 = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]
# capping the penalty weight keeps the merit function resolvable once χ reaches the solver
# accuracy; without the cap, w keeps growing and the run ends at `maxiter` as :Feasible
algo = SCPLib.SCvxStar(nx, N; w0 = 1e4, Δ0 = Δ0, l1_penalty = true, w_max = 1e6)

# solve problem
solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
    verbosity = 1, maxiter = 100, tol_opt = 1e-5, tol_feas = 1e-6)

# propagate solution
sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
Δv_norms = [norm(solution.u[2:4, k]) for k in 1:N]
println("status       = $(solution.status)")
println("total Δv     = $(objective(solution.x, solution.u)) (DU/TU)")
println("time of flight = $(solution.x[7, end]) (TU)")
println("max |g_dyn|  = $(maximum(abs, g_dynamics_opt))")


# -------------------- plot -------------------- #
fig = Figure(size=(1400, 800))

# trajectory: departure/arrival LPOs, transfer arcs (red after an impulse), impulse locations
ax3d = Axis3(fig[1, 1]; aspect=:data, xlabel="x", ylabel="y", zlabel="z")
lines!(ax3d, Array(sol_lpo0)[1, :], Array(sol_lpo0)[2, :], Array(sol_lpo0)[3, :], color=:blue)
lines!(ax3d, Array(sol_lpof)[1, :], Array(sol_lpof)[2, :], Array(sol_lpof)[3, :], color=:green)
arc_colors = [Δv_norms[i] > 1e-6 ? :red : :black for i in 1:N-1]
for (i, _sol) in enumerate(sols_opt)
    lines!(ax3d, Array(_sol)[1, :], Array(_sol)[2, :], Array(_sol)[3, :], color=arc_colors[i])
end
burn_nodes = findall(>(1e-6), Δv_norms)
scatter!(ax3d, solution.x[1, burn_nodes], solution.x[2, burn_nodes], solution.x[3, burn_nodes],
    color=:red, markersize=8)

# controls, against physical time: impulses at the nodes, dilation per segment
gl_u = fig[2, 1] = GridLayout()
t_nodes = solution.x[7, :]
ax_u = Axis(gl_u[1, 1]; ylabel="Impulse")
for i in 1:3
    stem!(ax_u, t_nodes, solution.u[1+i, :], label="Δv[$i]")
end
scatter!(ax_u, t_nodes, solution.u[5, :], label="Γ", color=:black, marker=:hline, markersize=14)
axislegend(ax_u, position=:cc)
ax_s = Axis(gl_u[2, 1]; xlabel="Time", ylabel="Dilation")
stairs!(ax_s, t_nodes, [solution.u[1, 1:N-1]; solution.u[1, N-1]], step=:post, color=:black)
linkxaxes!(ax_u, ax_s)
hidexdecorations!(ax_u, grid=false)

# iterate information (iterations with ΔJ = 0, i.e. rejected steps, are not shown on the log scale)
n_iter = length(solution.info[:accept])
colors_accept = [solution.info[:accept][i] ? :green : :red for i in 1:n_iter]
ax_χ = Axis(fig[1, 2]; xlabel="Iteration", ylabel="χ", yscale=log10)
scatterlines!(ax_χ, 1:n_iter, solution.info[:χ], color=colors_accept, marker=:circle, markersize=7)

ax_w = Axis(fig[2, 2]; xlabel="Iteration", ylabel="w", yscale=log10)
scatterlines!(ax_w, 1:n_iter, solution.info[:w], color=colors_accept, marker=:circle, markersize=7)

ax_J = Axis(fig[1, 3]; xlabel="Iteration", ylabel="ΔJ", yscale=log10)
abs_ΔJ = [v == 0 ? NaN : abs(v) for v in solution.info[:ΔJ]]
scatterlines!(ax_J, 1:n_iter, abs_ΔJ, color=colors_accept, marker=:circle, markersize=7)

ax_Δ = Axis(fig[2, 3]; xlabel="Iteration", ylabel="trust region radius", yscale=log10)
scatterlines!(ax_Δ, 1:n_iter, [minimum(val) for val in solution.info[:Δ]], color=colors_accept, marker=:circle, markersize=7)

save(joinpath(@__DIR__, "plots/cr3bp_impulsive_freetf_scvxstar.png"), fig; px_per_unit=3)
display(fig)
println("Done!")
