"""Benchmark SCvx* algorithm on the CR3BP problem from ex_scvxstar_cr3bp.jl"""

using BenchmarkTools
using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


struct ControlParams_benchmark
    μ::Float64
end

function eom!(drv, rv, pu, t)
    (; params, u) = pu
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt((x + params.μ)^2 + y^2 + z^2)
    r2 = sqrt((x - 1 + params.μ)^2 + y^2 + z^2)
    drv[1:3] = rv[4:6]
    drv[4] = 2 * vy + x - ((1 - params.μ) / r1^3) * (params.μ + x) + (params.μ / r2^3) * (1 - params.μ - x)
    drv[5] = -2 * vx + y - ((1 - params.μ) / r1^3) * y - (params.μ / r2^3) * y
    drv[6] = -((1 - params.μ) / r1^3) * z - (params.μ / r2^3) * z
    drv[4:6] += u[1:3]
    return
end

μ = 1.215058560962404e-02
DU = 389703
TU = 382981
MU = 500.0
VU = DU / TU

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

N = 60
nx = 6
nu = 4
tf = 2.6
umax = 0.35 / MU / 1e3 / (VU / TU)

objective(x, u) = sum(u[4, :])

function build_problem()
    params = ControlParams_benchmark(μ)
    times = LinRange(0.0, tf, N)

    sol_lpo0 = solve(
        ODEProblem(eom!, rv0, [0.0, period_0], (; params, u=zeros(nu))),
        Tsit5(); reltol=1e-12, abstol=1e-12,
    )
    sol_lpof = solve(
        ODEProblem(eom!, rvf, [0.0, period_f], (; params, u=zeros(nu))),
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
