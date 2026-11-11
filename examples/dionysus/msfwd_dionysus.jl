"""Max-final-mass rendezvous with asteroid Dionysus at a fixed time of flight.

Default multiple shooting with SCvx*. The state at every node is a decision
variable. Thrust uses the convex cone `||u[1:3]|| ≤ Γ ≤ 1`, with acceleration
`(c1 / m) * u[1:3]` and mass flow `ṁ = -c2 * Γ`. The node count matches
`TOPLib.jl/examples/dionysus/msfwd_dionysus.jl`.

Run from this directory:

    julia -t auto --project=.. msfwd_dionysus.jl
"""

using CairoMakie
using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

import AstrodynamicsCore: Planet, eph, kep2rv, ma2ta, mee2rv, rv2mee

include(joinpath(@__DIR__, "../../src/SCPLib.jl"))


struct ControlParams
    μ::Float64
    c1::Float64
    c2::Float64
end

const MU_SUN = 132712440018.0
const G0 = 9.8065
const DU = 149.6e6                 # km
const VU = sqrt(MU_SUN / DU)       # km/s
const TU = DU / VU                 # s
const MASS = 4000.0                # kg
const THRUST = 0.32                # N
const ISP = 3000.0                 # s

μ = MU_SUN / (VU^2 * DU)
c1 = THRUST / 1e3 / (MASS * DU / TU^2)
c2 = THRUST / 1e3 / (ISP * G0 / 1e3) / (MASS / TU)
params = ControlParams(μ, c1, c2)

function eom_mee!(drvm, rvm, pu, t)
    (; params, u) = pu
    p, f, g, h, k, L, mass = rvm[1:7]
    cosL = cos(L)
    sinL = sin(L)
    s2 = 1 + h^2 + k^2
    w = 1 + f * cosL + g * sinL
    hsinL_kcosL = h * sinL - k * cosL
    B_mee = sqrt(p / params.μ) * [
        0     2p / w                    0;
        sinL  ((1 + w) * cosL + f) / w  -g / w * hsinL_kcosL;
        -cosL ((1 + w) * sinL + g) / w   f / w * hsinL_kcosL;
        0     0                          1 / w * s2 / 2 * cosL;
        0     0                          1 / w * s2 / 2 * sinL;
        0     0                          1 / w * hsinL_kcosL
    ]
    D = [0.0, 0.0, 0.0, 0.0, 0.0, sqrt(params.μ / p^3) * w^2]
    drvm[1:6] = B_mee * (params.c1 / mass) * u[1:3] + D
    drvm[7] = -u[4] * params.c2
    return nothing
end

tof = 3534 * 86400 / TU
N_rev = 5

R0 = [-3637871.081; 147099798.784; -2261.44] / DU
V0 = [-30.265097; -0.8486854; 0.0000505] / VU
RV0 = [R0; V0]
orbit_init = Planet(params.μ, 0.0, RV0, "initial")
mee0 = rv2mee(RV0, params.μ)

M0 = deg2rad(114.4232)
TA0 = ma2ta(M0, 0.542)
kepf = [2.2, 0.542, deg2rad(13.6), deg2rad(82.2), deg2rad(204.2), TA0]
RVf0 = kep2rv(kepf, params.μ)
orbit_final = Planet(params.μ, 0.0, RVf0, "final")
RVf = eph(orbit_final, tof + (56284 - 53400) * 86400 / TU)
meef = rv2mee(RVf, params.μ)
meef[6] += 2π * N_rev

initial_orbit_rvs = hcat([eph(orbit_init, t) for t in LinRange(0.0, orbit_init.period, 300)]...)
final_orbit_rvs = hcat([eph(orbit_final, t) for t in LinRange(0.0, orbit_final.period, 300)]...)

N = 41
nx = 7
nu = 4
ng = 6
times = LinRange(0.0, tof, N)

x0_ref = [mee0; 1.0]
x_ref = zeros(nx, N)
x_ref[1:6, :] = hcat(LinRange.(mee0, meef, N)...)'
x_ref[7, :] = LinRange(1.0, 0.4, N)
u_ref = zeros(nu, N - 1)

objective(x, u) = -x[7, end]

function g_noncvx(lincache, x, u)
    return mee2rv(x[1:6, end], params.μ) - RVf
end

prob = SCPLib.ContinuousProblem(
    Clarabel.Optimizer,
    eom_mee!,
    params,
    objective,
    times,
    x_ref,
    u_ref;
    ng=ng,
    g_noncvx=g_noncvx,
    ode_ensemble_method=EnsembleThreads(),
    ode_method=Vern7(),
)
set_silent(prob.model)

@constraint(prob.model, constraint_initial_rv, prob.model[:x][:, 1] == x0_ref)
@constraint(prob.model, constraint_monotonic_L[k in 1:N-1], prob.model[:x][6, k] <= prob.model[:x][6, k+1])
@constraint(prob.model, constraint_mass_lb[k in 1:N], prob.model[:x][7, k] >= 0.1)
@constraint(prob.model, constraint_p_lb[k in 1:N], prob.model[:x][1, k] >= 0.8)
@constraint(prob.model, constraint_associate_control[k in 1:N-1],
    [prob.model[:u][4, k], prob.model[:u][1:3, k]...] in SecondOrderCone())
@constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
    prob.model[:u][4, k] <= 1.0)

# Thrust enters the equinoctial dynamics nonlinearly, so the control step is bounded too.
algo = SCPLib.SCvxStar(nx, N; ng=ng, w0=1e0, w_max=1e20, use_trustregion_control=true)
solution = SCPLib.solve!(
    algo, prob, x_ref, u_ref;
    tol_opt=1e-6, tol_feas=1e-6, maxiter=200,
)

sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
rv_err = mee2rv(solution.x[1:6, end], params.μ) - RVf
println("status           : ", solution.status)
println("iterations       : ", solution.n_iter)
println("final mass       : ", solution.x[7, end] * MASS, " kg")
println("max |defect|     : ", maximum(abs, g_dynamics_opt))
println("max |rendezvous| : ", maximum(abs, rv_err))

fig = Figure(size=(1000, 800))
ax3d = Axis3(fig[1, 1]; aspect=:data, xlabel="x, AU", ylabel="y, AU", zlabel="z, AU")
rowsize!(fig.layout, 1, Relative(0.62))
scatter!(ax3d, RV0[1], RV0[2], RV0[3], color=:limegreen, markersize=10)
scatter!(ax3d, RVf[1], RVf[2], RVf[3], color=:blue, markersize=10)
lines!(ax3d, initial_orbit_rvs[1, :], initial_orbit_rvs[2, :], initial_orbit_rvs[3, :], color=:limegreen, linewidth=0.8)
lines!(ax3d, final_orbit_rvs[1, :], final_orbit_rvs[2, :], final_orbit_rvs[3, :], color=:blue, linewidth=0.8)
for (isol, sol) in enumerate(sols_opt)
    mee = Array(sol)
    rvs = hcat([mee2rv(mee[1:6, j], params.μ) for j in axes(mee, 2)]...)
    lines!(ax3d, rvs[1, :], rvs[2, :], rvs[3, :], color=solution.u[4, isol] > 1e-2 ? :red : :black)
end

ax_u = Axis(fig[1, 2]; xlabel="Time, day", ylabel="Control")
hlines!(ax_u, 1.0, color=:red, linestyle=:dash)
t_days = collect(times) .* TU ./ 86400
control_labels = ["u_r", "u_θ", "u_n", "Γ"]
for i in 1:nu
    stairs!(ax_u, t_days, [solution.u[i, :]; 0.0], label=control_labels[i], step=:post, linewidth=1.0)
end
Legend(fig[1, 3], ax_u)

ax_m = Axis(fig[2, 1:3]; xlabel="Time, day", ylabel="Mass, kg")
for (isol, sol) in enumerate(sols_opt)
    mee = Array(sol)
    lines!(ax_m, sol.t .* TU ./ 86400, mee[7, :] .* MASS, color=solution.u[4, isol] > 1e-2 ? :red : :black)
end

plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)
plot_path = joinpath(plots_dir, "dionysus_msfwd.png")
save(plot_path, fig; px_per_unit=3)
println("Saved plot to ", plot_path)
