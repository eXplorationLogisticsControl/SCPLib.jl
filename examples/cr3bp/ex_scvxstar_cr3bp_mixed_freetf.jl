"""Variable-time CR3BP transfer with mixed continuous and impulsive controls.

State is `[rx, ry, rz, vx, vy, vz, t_phys]`. The independent variable is
`LinRange(0, 1, N)`. The only continuous control is a per-segment time
dilation. The impulsive control is `[Δvx, Δvy, Δvz, Γ]`, including an
arrival burn at node `N`. The objective is the sum of `Γ`.
"""

using CairoMakie
using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../../src/SCPLib.jl"))


struct ControlParams
    μ::Float64
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU / TU    # km/s
params = ControlParams(μ)

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

# Coast: dilation times [f_cr3bp(r, v); 1]. `u` is the continuous slice only.
function eom!(dx, x, pu, t)
    (; params, u) = pu
    cr3bp_field!(dx, x, params.μ)
    dx[1:7] .*= u[1]
    return
end

# Velocity jump from the first three impulsive entries. Slack and time stay put.
function dfdu(x, u_i, t)
    return [zeros(3, 4); I(3) zeros(3, 1); zeros(1, 4)]
end

# User augmented coast: Φ_A and the continuous Φ_B (width nuc = 1) only.
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

function objective(x, u)
    return sum(u[size(u, 1), k] for k in 1:size(u, 2))
end

# Same Lyapunov orbits as the other CR3BP examples.
rv0 = [
    1.0809931218390707E+00,
    0.0000000000000000E+00,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15,
]
period_0 = 2.3538670417546639E+00
rvf = [
    1.1648780946517576,
    0.0,
    -1.1145303634437023E-1,
    0.0,
    -2.0191923237095796E-1,
    0.0,
]
period_f = 3.3031221822879884

coast! = function (dx, x, pu, t)
    cr3bp_field!(dx, x, μ)
    dx[7] = 0.0
end
sol_lpo0 = solve(ODEProblem(coast!, [rv0; 0.0], (0.0, period_0), nothing), Tsit5(); reltol = 1e-12, abstol = 1e-12)
sol_lpof = solve(ODEProblem(coast!, [rvf; 0.0], (0.0, period_f), nothing), Tsit5(); reltol = 1e-12, abstol = 1e-12)

N = 21
nx = 7
nuc = 1
nui = 4
nu = nuc + nui
times = LinRange(0.0, 1.0, N)
tf_guess = 3.0
tf_span = [2.0, 4.0]   # copied from ex_scvxstar_cr3bp_mass_freetf_ad.jl

x_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
x_lpof = sol_lpof(LinRange(0.0, period_f, N))
x_ref = zeros(nx, N)
for (i, α) in enumerate(LinRange(0.0, 1.0, N))
    x_ref[1:6, i] = (1 - α) * x_lpo0[1:6, i] + α * x_lpof[1:6, i]
end
x_ref[7, :] = LinRange(0.0, tf_guess, N)

u_ref = zeros(nu, N)
u_ref[1, 1:N-1] .= tf_guess
u_ref[1, N] = NaN

thrust = 0.35    # N
umax = thrust / MU / 1e3 / (VU / TU)

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

# Epoch is pinned so the final clock is the flight time. Final time and each
# dilation use the same bounds as the mass free-final-time example. Dilation
# is not forced equal across segments.
@constraint(prob.model, constraint_initial_rv, prob.model[:x][1:6, 1] .== rv0)
@constraint(prob.model, constraint_initial_time, prob.model[:x][7, 1] == 0.0)
@constraint(prob.model, constraint_final_r, prob.model[:x][1:3, end] .== rvf[1:3])
@constraint(prob.model, constraint_final_v, prob.model[:x][4:6, end] + prob.model[:ui][1:3, end] .== rvf[4:6])
@constraint(prob.model, constraint_tscale_lb[k in 1:N-1], prob.model[:uc][1, k] >= tf_span[1])
@constraint(prob.model, constraint_tscale_ub[k in 1:N-1], prob.model[:uc][1, k] <= tf_span[2])
@constraint(prob.model, constraint_tf_lb, prob.model[:x][7, end] >= tf_span[1])
@constraint(prob.model, constraint_tf_ub, prob.model[:x][7, end] <= tf_span[2])
@constraint(prob.model, constraint_associate_control[k in 1:N],
    [prob.model[:ui][4, k], prob.model[:ui][1:3, k]...] in SecondOrderCone())
@constraint(prob.model, constraint_control_magnitude[k in 1:N],
    prob.model[:ui][4, k] <= umax)

Δ0 = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]
algo = SCPLib.SCvxStar(nx, N; w0 = 1e4, Δ0 = Δ0, l1_penalty = true)
solution = SCPLib.solve!(
    algo, prob, x_ref, u_ref;
    verbosity = 1, maxiter = 80, tol_opt = 1e-5, tol_feas = 1e-6,
)

println("status = $(solution.status)")
println("J      = $(objective(solution.x, solution.u))")
println("tf     = $(solution.x[7, end])")

sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
println("max |g| = $(maximum(abs, g_dynamics_opt))")

fig = Figure(size=(1100, 700))
ax3d = Axis3(fig[1, 1]; aspect=:data, xlabel="x", ylabel="y", zlabel="z")
lines!(ax3d, Array(sol_lpo0)[1, :], Array(sol_lpo0)[2, :], Array(sol_lpo0)[3, :], color=:blue, label="departure LPO")
lines!(ax3d, Array(sol_lpof)[1, :], Array(sol_lpof)[2, :], Array(sol_lpof)[3, :], color=:green, label="arrival LPO")
for (i, _sol) in enumerate(sols_opt)
    color = solution.u[5, i] > 1e-6 ? :red : :black
    lines!(ax3d, Array(_sol)[1, :], Array(_sol)[2, :], Array(_sol)[3, :], color=color, linewidth=1.5)
end
axislegend(ax3d, position=:lt)

ax_u = Axis(fig[2, 1]; xlabel="physical time", ylabel="Γ")
stem!(ax_u, solution.x[7, :], solution.u[5, :], color=:black)

ax_s = Axis(fig[1, 2]; xlabel="physical time", ylabel="dilation")
t_s = Float64[]
s_s = Float64[]
for (i, _sol) in enumerate(sols_opt)
    append!(t_s, Array(_sol)[7, :])
    append!(s_s, fill(solution.u[1, i], length(_sol.t)))
end
stairs!(ax_s, t_s, s_s, step=:pre, color=:black)

colors_accept = [solution.info[:accept][i] ? :green : :red for i in 1:length(solution.info[:accept])]
ax_χ = Axis(fig[2, 2]; xlabel="iteration", ylabel="χ", yscale=log10)
scatterlines!(ax_χ, 1:length(solution.info[:χ]), solution.info[:χ], color=colors_accept, marker=:circle, markersize=6)

save(joinpath(@__DIR__, "plots/cr3bp_mixed_freetf_scvxstar.png"), fig)
display(fig)
println("Done!")
