"""Mixed continuous/impulsive problem: indexing, linearization, and a variable-time CR3BP transfer."""

using Clarabel
using ForwardDiff
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end
if !@isdefined verbosity
    verbosity = 0
end


struct ToyMixedParams end


function toy_eom!(dx, x, pu, t)
    s = pu.u[1]
    dx[1] = s * x[2]
    dx[2] = 0.0
    dx[3] = s
    return
end


function toy_dfdu(x, u_i, t)
    return [0.0 0.0; 1.0 0.0; 0.0 0.0]
end


function toy_eom_aug!(dx, xaug, pu, t)
    s = pu.u[1]
    x = xaug[1:3]
    dx[1] = s * x[2]
    dx[2] = 0.0
    dx[3] = s
    A = [0.0 s 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    B = [x[2], 0.0, 1.0]
    Φ = reshape(@view(xaug[4:12]), 3, 3)
    ΦB = @view(xaug[13:15])
    dx[4:12] = reshape(A * Φ, 9)
    dx[13:15] = A * ΦB + B
    return
end


function toy_objective(x, u)
    return sum(u[size(u, 1), k] for k in 1:size(u, 2))
end


function toy_reference()
    N = 3
    nuc, nui = 1, 2
    times = [0.0, 0.5, 1.0]
    # Feasible coast: Δv = 1 at node 1, s = 1, terminal Δv = -1.
    x_ref = [0.0 0.5 1.0;
             0.0 1.0 1.0;
             0.0 0.5 1.0]
    u_ref = [1.0 1.0 NaN;
             1.0 0.0 -1.0;
             1.0 0.0 1.0]
    return nuc, nui, times, x_ref, u_ref
end


function make_toy(; eom_aug! = nothing, ng::Int = 0, g_noncvx = nothing, u_bias = nothing)
    nuc, nui, times, x_ref, u_ref = toy_reference()
    prob = SCPLib.MixedProblem(
        Clarabel.Optimizer,
        toy_eom!,
        ToyMixedParams(),
        toy_objective,
        times,
        x_ref,
        u_ref;
        nuc = nuc,
        nui = nui,
        dfdu = toy_dfdu,
        eom_aug! = eom_aug!,
        ng = ng,
        g_noncvx = g_noncvx,
        u_bias = u_bias,
        ode_method = Tsit5(),
    )
    set_silent(prob.model)
    return prob, x_ref, u_ref
end


function test_constructor_requires_nan_arrival_column()
    nuc, nui, times, x_ref, u_ref = toy_reference()
    u_bad = copy(u_ref)
    u_bad[1, end] = 0.0
    @test_throws AssertionError SCPLib.MixedProblem(
        Clarabel.Optimizer, toy_eom!, ToyMixedParams(), toy_objective,
        times, x_ref, u_bad;
        nuc = nuc, nui = nui, dfdu = toy_dfdu,
    )
end


function test_split_variables_and_objective()
    prob, x_ref, u_ref = make_toy()
    @test prob.nuc == 1 && prob.nui == 2 && prob.nu == 3 && prob.N == 3 && prob.Nu == 3
    @test size(prob.model[:uc]) == (1, 2)
    @test size(prob.model[:ui]) == (2, 3)
    @test isnan(prob.model[:u][1, 3])
    @test prob.model[:u][2, 3] === prob.model[:ui][1, 3]
    @test prob.model[:u][1, 1] === prob.model[:uc][1, 1]
    J = prob.objective(prob.model[:x], prob.model[:u])
    @test J isa JuMP.AbstractJuMPScalar
    @test prob.objective(x_ref, u_ref) ≈ 2.0
    u_poison = copy(u_ref)
    u_poison[1, end] = 1e9
    @test prob.objective(x_ref, u_poison) ≈ 2.0
end


function test_flatten_skips_arrival_continuous_column()
    prob, x_ref, u_ref = make_toy()
    z = SCPLib.stack_flatten_variables(prob, x_ref, u_ref)
    @test length(z) == prob.nx * prob.N + prob.nuc * (prob.N - 1) + prob.nui * prob.N
    @test all(isfinite, z)
    x2, u2 = SCPLib.unpack_flattened_variables(prob, z)
    @test x2 ≈ x_ref
    @test u2[1:1, 1:2] ≈ u_ref[1:1, 1:2]
    @test u2[2:3, :] ≈ u_ref[2:3, :]
    @test isnan(u2[1, end])
end


function test_propagation_ignores_arrival_continuous_entry()
    prob, x_ref, u_ref = make_toy()
    _, g1 = SCPLib.get_trajectory(prob, x_ref, u_ref)
    @test norm(g1, Inf) < 1e-8
    u_poison = copy(u_ref)
    u_poison[1, end] = 1e9
    _, g2 = SCPLib.get_trajectory(prob, x_ref, u_poison)
    @test g1 ≈ g2
end


function test_phi_splits_continuous_and_impulsive_blocks()
    prob_ad, x_ref, u_ref = make_toy()
    prob_user, _, _ = make_toy(; eom_aug! = toy_eom_aug!)
    SCPLib.set_dynamics_cache!(prob_ad, x_ref, u_ref)
    SCPLib.set_dynamics_cache!(prob_user, x_ref, u_ref)

    # One segment, Δτ = 0.5, s = 1, v_after = 1.
    # Φ_A = [1 s*Δτ 0; 0 1 0; 0 0 1]
    # Φ_Bc = Δτ * [v_after, 0, 1]
    # Φ_Bi = Φ_A * dfdu
    s, Δτ, v_after = 1.0, 0.5, 1.0
    Φ_A = [1.0 s*Δτ 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Φ_Bc = Δτ * [v_after, 0.0, 1.0]
    Φ_Bi = Φ_A * toy_dfdu(x_ref[:, 1], u_ref[2:3, 1], 0.0)
    for prob in (prob_ad, prob_user)
        @test prob.lincache.Φ_A[:, :, 1] ≈ Φ_A atol = 1e-8
        @test prob.lincache.Φ_B[:, 1, 1] ≈ Φ_Bc atol = 1e-8
        @test prob.lincache.Φ_B[:, 2:3, 1] ≈ Φ_Bi atol = 1e-8
        u_k = u_ref[:, 1]
        x_end = prob.lincache.Φ_A[:, :, 1] * x_ref[:, 1] + prob.lincache.Φ_B[:, :, 1] * u_k + prob.lincache.Φ_c[:, 1]
        @test x_end ≈ x_ref[:, 2] atol = 1e-8
    end
end


function test_nonconvex_jacobian_excludes_nan_column()
    g_noncvx(cache, x, u) = [x[1, 2] - 0.5]
    prob, x_ref, u_ref = make_toy(; ng = 1, g_noncvx = g_noncvx)
    nflat = prob.nx * prob.N + prob.nuc * (prob.N - 1) + prob.nui * prob.N
    @test size(prob.lincache.∇g) == (1, nflat)
    _, g_ref, _ = SCPLib.set_linearized_constraints!(prob, x_ref, u_ref)
    @test g_ref ≈ [0.0] atol = 1e-10
    # z = [x (column major); for k=1:2: s, Δv, Γ; then terminal Δv, Γ]
    # x[1,2] is index 4 (x11, x21, x31, x12, ...)
    ∇ = prob.lincache.∇g
    @test ∇[1, 4] ≈ 1.0
    @test count(!iszero, ∇) == 1
    @test all(isfinite, ∇)
end


function test_trust_region_skips_arrival_continuous_column()
    prob, x_ref, u_ref = make_toy()
    algo = SCPLib.SCvxStar(prob.nx, prob.N; nu = prob.nu, Δ0_u = 0.2, use_trustregion_control = true, w0 = 10.0)
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity = 0, maxiter = 5, tol_feas = 1e-6, tol_opt = 1e-4)
    @test isnan(algo.tr_u.Δ[1, prob.N])
    @test all(isfinite, algo.tr_u.Δ[1, 1:prob.N-1])
    @test all(isfinite, algo.tr_u.Δ[2:3, :])
    @test isnan(solution.u[1, end])
    @test all(isfinite, solution.u[2:3, :])
    @test all(isfinite, solution.u[1, 1:end-1])
    @test solution.status in (:Optimal, :Feasible, :Solving)
end


function test_toy_transfer_solves()
    prob, x_ref, u_ref = make_toy()
    for k in 1:prob.N-1
        @constraint(prob.model, prob.model[:uc][1, k] >= 0.5)
        @constraint(prob.model, prob.model[:uc][1, k] <= 2.0)
    end
    @constraint(prob.model, prob.model[:x][:, 1] .== x_ref[:, 1])
    @constraint(prob.model, prob.model[:x][1, end] == 1.0)
    @constraint(prob.model, prob.model[:x][2, end] + prob.model[:ui][1, end] == 0.0)
    @constraint(prob.model, prob.model[:x][3, end] >= 0.5)
    @constraint(prob.model, prob.model[:x][3, end] <= 2.0)
    for k in 1:prob.N
        @constraint(prob.model, [prob.model[:ui][2, k], prob.model[:ui][1, k]] in SecondOrderCone())
    end
    algo = SCPLib.SCvxStar(prob.nx, prob.N; w0 = 50.0)
    solution = SCPLib.solve!(algo, prob, copy(x_ref), copy(u_ref); verbosity = 0, maxiter = 30, tol_feas = 1e-6, tol_opt = 1e-5)
    @test solution.status == :Optimal
    @test isnan(solution.u[1, end])
    _, g = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @test norm(g, Inf) <= 1e-6
    @test solution.x[1, end] ≈ 1.0 atol = 1e-5
    @test solution.x[2, end] + solution.u[2, end] ≈ 0.0 atol = 1e-5
    @test toy_objective(solution.x, solution.u) ≈ sum(solution.u[3, :])
end


function test_other_solvers_keep_split_controls()
    makers = [
        () -> SCPLib.SCvx(3, 3; w = 10.0),
        () -> SCPLib.ProxLinear(10.0, 1.0; proximal_u = true),
        () -> SCPLib.FixedTRWSCP(3, 3, 0.5, 10.0),
    ]
    for maker in makers
        prob, x_ref, u_ref = make_toy()
        solution = SCPLib.solve!(maker(), prob, copy(x_ref), copy(u_ref); verbosity = 0, maxiter = 3)
        @test isnan(solution.u[1, end])
        @test all(isfinite, solution.u[1, 1:end-1])
        @test all(isfinite, solution.u[2:3, :])
        @test size(solution.u) == size(u_ref)
    end
end


# -------------------- variable-time CR3BP -------------------- #

struct CR3BPMixedParams
    μ::Float64
end

const CR3BP_RV0 = [
    1.0809931218390707E+00,
    0.0,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15,
]
const CR3BP_PERIOD0 = 2.3538670417546639E+00
const CR3BP_RVF = [
    1.1648780946517576,
    0.0,
    -1.1145303634437023E-1,
    0.0,
    -2.0191923237095796E-1,
    0.0,
]
const CR3BP_PERIODF = 3.3031221822879884
const CR3BP_TF_SPAN = [2.0, 4.0]


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


function cr3bp_mixed_eom!(dx, x, pu, t)
    (; params, u) = pu
    cr3bp_field!(dx, x, params.μ)
    dx[1:7] .*= u[1]
    return
end


"""7×4 jump: velocity from the first three impulsive entries, slack and time unchanged."""
function cr3bp_mixed_dfdu(x, u_i, t)
    return [zeros(3, 4); I(3) zeros(3, 1); zeros(1, 4)]
end


function cr3bp_mixed_eom_aug!(dx, xaug, pu, t)
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


function cr3bp_objective(x, u)
    return sum(u[size(u, 1), k] for k in 1:size(u, 2))
end


function make_cr3bp_mixed(; N::Int = 21, eom_aug! = cr3bp_mixed_eom_aug!)
    μ = 1.215058560962404e-02
    params = CR3BPMixedParams(μ)
    nuc, nui = 1, 4
    nx = 7
    nu = nuc + nui
    times = LinRange(0.0, 1.0, N)
    tf_guess = 3.0

    # Geometry of the two Lyapunov orbits, sampled on their own periods.
    coast! = function (dx, x, pu, t)
        cr3bp_field!(dx, x, μ)
        dx[7] = 0.0
    end
    x0 = [CR3BP_RV0; 0.0]
    xf = [CR3BP_RVF; 0.0]
    sol0 = solve(ODEProblem(coast!, x0, (0.0, CR3BP_PERIOD0), (; params, u = [1.0])), Tsit5(); reltol = 1e-12, abstol = 1e-12)
    solf = solve(ODEProblem(coast!, xf, (0.0, CR3BP_PERIODF), (; params, u = [1.0])), Tsit5(); reltol = 1e-12, abstol = 1e-12)
    x_lpo0 = sol0(LinRange(0.0, CR3BP_PERIOD0, N))
    x_lpof = solf(LinRange(0.0, CR3BP_PERIODF, N))

    x_ref = zeros(nx, N)
    for (i, α) in enumerate(LinRange(0.0, 1.0, N))
        x_ref[1:6, i] = (1 - α) * x_lpo0[1:6, i] + α * x_lpof[1:6, i]
    end
    x_ref[7, :] = LinRange(0.0, tf_guess, N)

    u_ref = zeros(nu, N)
    u_ref[1, 1:N-1] .= tf_guess
    u_ref[1, N] = NaN

    prob = SCPLib.MixedProblem(
        Clarabel.Optimizer,
        cr3bp_mixed_eom!,
        params,
        cr3bp_objective,
        times,
        x_ref,
        u_ref;
        nuc = nuc,
        nui = nui,
        dfdu = cr3bp_mixed_dfdu,
        eom_aug! = eom_aug!,
        ode_method = Vern7(),
    )
    set_silent(prob.model)

    DU, TU, MU = 389703.0, 382981.0, 500.0
    VU = DU / TU
    umax = 0.35 / MU / 1e3 / (VU / TU)

    @constraint(prob.model, prob.model[:x][1:6, 1] .== CR3BP_RV0)
    @constraint(prob.model, prob.model[:x][7, 1] == 0.0)
    @constraint(prob.model, prob.model[:x][1:3, end] .== CR3BP_RVF[1:3])
    @constraint(prob.model, prob.model[:x][4:6, end] + prob.model[:ui][1:3, end] .== CR3BP_RVF[4:6])
    for k in 1:N-1
        @constraint(prob.model, prob.model[:uc][1, k] >= CR3BP_TF_SPAN[1])
        @constraint(prob.model, prob.model[:uc][1, k] <= CR3BP_TF_SPAN[2])
    end
    @constraint(prob.model, prob.model[:x][7, end] >= CR3BP_TF_SPAN[1])
    @constraint(prob.model, prob.model[:x][7, end] <= CR3BP_TF_SPAN[2])
    for k in 1:N
        @constraint(prob.model, [prob.model[:ui][4, k]; prob.model[:ui][1:3, k]] in SecondOrderCone())
        @constraint(prob.model, prob.model[:ui][4, k] <= umax)
    end
    return prob, x_ref, u_ref
end


function test_cr3bp_autodiff_aug_matches_analytic()
    # Short horizon: compare one linearization, not a full solve.
    prob_ad, x_ref, u_ref = make_cr3bp_mixed(; N = 3, eom_aug! = nothing)
    prob_an, _, _ = make_cr3bp_mixed(; N = 3, eom_aug! = cr3bp_mixed_eom_aug!)
    # Rebuild without the constraints duplicating names: make_cr3bp_mixed adds constraints.
    # The two problems are independent models, so this is fine.
    SCPLib.set_dynamics_cache!(prob_ad, x_ref, u_ref)
    SCPLib.set_dynamics_cache!(prob_an, x_ref, u_ref)
    @test prob_ad.lincache.Φ_A ≈ prob_an.lincache.Φ_A atol = 1e-6
    @test prob_ad.lincache.Φ_B ≈ prob_an.lincache.Φ_B atol = 1e-6
    B = cr3bp_mixed_dfdu(x_ref[:, 1], u_ref[2:5, 1], 0.0)
    @test size(B) == (7, 4)
    @test B[1:3, :] == zeros(3, 4)
    @test B[4:6, 1:3] == I(3)
    @test B[4:6, 4] == zeros(3)
    @test B[7, :] == zeros(4)
    @test prob_an.lincache.Φ_B[:, 2:5, 1] ≈ prob_an.lincache.Φ_A[:, :, 1] * B atol = 1e-6
end


function test_cr3bp_variable_time_transfer(; verbosity::Int = 0)
    N = 21
    prob, x_ref, u_ref = make_cr3bp_mixed(; N = N)
    Δ0 = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1]
    algo = SCPLib.SCvxStar(prob.nx, prob.N; w0 = 1e4, Δ0 = Δ0, l1_penalty = true)
    solution = SCPLib.solve!(
        algo, prob, x_ref, u_ref;
        verbosity = verbosity, maxiter = 80, tol_opt = 1e-5, tol_feas = 1e-6,
    )
    # The transfer cost settles while the augmented-Lagrangian weight is still
    # climbing, so the penalized ΔJ stays above tol_opt and the exit status is
    # :Feasible. The checks below are on the nonlinear transfer, not that flag.
    @test solution.status in (:Optimal, :Feasible)
    @test solution.info[:χ][end] <= 1e-6
    @test isnan(solution.u[1, end])
    @test all(isfinite, solution.u[1, 1:end-1])
    @test all(isfinite, solution.u[2:5, :])
    _, g = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @test norm(g, Inf) <= 1e-6
    @test solution.x[1:3, end] ≈ CR3BP_RVF[1:3] atol = 1e-5
    @test solution.x[4:6, end] + solution.u[2:4, end] ≈ CR3BP_RVF[4:6] atol = 1e-5
    @test solution.x[7, 1] ≈ 0.0 atol = 1e-6
    @test CR3BP_TF_SPAN[1] - 1e-5 <= solution.x[7, end] <= CR3BP_TF_SPAN[2] + 1e-5
    @test all(solution.u[1, 1:end-1] .>= CR3BP_TF_SPAN[1] - 1e-5)
    @test all(solution.u[1, 1:end-1] .<= CR3BP_TF_SPAN[2] + 1e-5)
    cost = cr3bp_objective(solution.x, solution.u)
    @test cost ≈ sum(solution.u[end, :])
    @test cost ≈ 0.19225 atol = 1e-3
    for k in 1:N
        @test solution.u[5, k] + 1e-7 >= norm(solution.u[2:4, k])
    end
end


test_constructor_requires_nan_arrival_column()
test_split_variables_and_objective()
test_flatten_skips_arrival_continuous_column()
test_propagation_ignores_arrival_continuous_entry()
test_phi_splits_continuous_and_impulsive_blocks()
test_nonconvex_jacobian_excludes_nan_column()
test_trust_region_skips_arrival_continuous_column()
test_toy_transfer_solves()
test_other_solvers_keep_split_controls()
test_cr3bp_autodiff_aug_matches_analytic()
test_cr3bp_variable_time_transfer(; verbosity = verbosity)
