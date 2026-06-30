"""Benchmark SCvx* algorithm on the CR3BP problem from ex_scvxstar_cr3bp.jl"""

using BenchmarkTools
using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end

function eom!(drv, rv, p, t)
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt((x + p.μ)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + p.μ)^2 + y^2 + z^2)
    drv[1:3] = rv[4:6]
    drv[4] = 2 * vy + x - ((1 - p.μ) / r1^3) * (p.μ + x) + (p.μ / r2^3) * (1 - p.μ - x)
    drv[5] = -2 * vx + y - ((1 - p.μ) / r1^3) * y - (p.μ / r2^3) * y
    drv[6] = -((1 - p.μ) / r1^3) * z - (p.μ / r2^3) * z
    drv[4:6] += p.u[1:3]
    return
end

const μ = 1.215058560962404e-02
const DU = 389703
const TU = 382981
const MU = 500.0
const VU = DU / TU

const rv0 = [
    1.0809931218390707E+00,
    0.0000000000000000E+00,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15,
]
const period_0 = 2.3538670417546639E+00

const rvf = [
    1.1648780946517576,
    0.0,
    -1.1145303634437023E-1,
    0.0,
    -2.0191923237095796E-1,
    0.0,
]
const period_f = 3.3031221822879884

const N = 100
const nx = 6
const nu = 4
const tf = 2.6
const umax = 0.35 / MU / 1e3 / (VU / TU)

objective(x, u) = sum(u[4, :])

function build_problem()
    params = ControlParams(μ)
    times = LinRange(0.0, tf, N)

    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], params),
        Tsit5(); reltol=1e-12, abstol=1e-12,
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], params),
        Tsit5(); reltol=1e-12, abstol=1e-12,
    )

    x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
    x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
    x_ref_init = zeros(nx, N)
    alphas = LinRange(0, 1, N)
    for (i, alpha) in enumerate(alphas)
        x_ref_init[:, i] = (1 - alpha) * x_along_lpo0[:, i] + alpha * x_along_lpof[:, i]
    end
    u_ref_init = zeros(nu, N - 1)

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        params,
        objective,
        times,
        x_ref_init,
        u_ref_init;
        ode_method=Vern7(),
    )
    set_silent(prob.model)

    @constraint(prob.model, constraint_initial_rv, prob.model[:x][:, 1] == rv0)
    @constraint(prob.model, constraint_final_rv, prob.model[:x][:, end] == rvf)
    @constraint(prob.model, constraint_associate_control[k in 1:N-1],
        [prob.model[:u][4, k], prob.model[:u][1:3, k]...] in SecondOrderCone())
    @constraint(prob.model, constraint_control_magnitude[k in 1:N-1],
        prob.model[:u][4, k] <= umax)

    return prob, x_ref_init, u_ref_init
end

function make_solve_state()
    prob, x_ref_init, u_ref_init = build_problem()
    x_ref = copy(x_ref_init)
    u_ref = copy(u_ref_init)
    algo = SCPLib.SCvxStar(nx, N; w0=1e4)
    reset_w = Ref(false)
    callback = function (algo, solution, iteration, J0, χ)
        if iteration <= 30 && χ <= 1e-6 && !reset_w[]
            algo.w = 1e2
            reset_w[] = true
        end
    end
    return (prob=prob, x_ref=x_ref, u_ref=u_ref, algo=algo, callback=callback)
end

function run_solve!(state)
    return SCPLib.solve!(state.algo, state.prob, state.x_ref, state.u_ref;
        maxiter=100, callback=state.callback, verbosity=0)
end


println("Running warmup solve...")
warmup_state = make_solve_state()
warmup_solution = run_solve!(warmup_state)
println("Warmup status: ", warmup_solution.status, " (", warmup_solution.n_iter, " iterations)")

solve_state = Ref{Any}(nothing)
println("Running benchmark...")
bench = @benchmarkable begin
    state = $solve_state[]
    SCPLib.solve!(state.algo, state.prob, state.x_ref, state.u_ref;
        maxiter=100, callback=state.callback, verbosity=0)
end setup=(
    solve_state[] = make_solve_state()
)
params = BenchmarkTools.Parameters(samples=5, seconds=60, evals=1, evals_set=true)
result = BenchmarkTools.run(bench, params)

outdir = joinpath(@__DIR__, "out")
mkpath(outdir)
outfile = joinpath(outdir, "benchmark_scvxstar.txt")

open(outfile, "w") do io
    println(io, "SCvx* CR3BP benchmark")
    println(io, "Source: examples/ex_scvxstar_cr3bp.jl")
    println(io, "Julia: ", VERSION)
    println(io, "N = ", N, ", nx = ", nx, ", nu = ", nu)
    println(io, "Warmup status: ", warmup_solution.status, " (", warmup_solution.n_iter, " iterations)")
    println(io, "Note: each timed sample rebuilds the JuMP problem in setup (not timed).")
    println(io)
    show(io, MIME("text/plain"), result)
end

println("Wrote benchmark to ", outfile)
