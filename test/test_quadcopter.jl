"""Test SCvx* on the packaged quadcopter problem"""

using Clarabel
using LinearAlgebra

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end


function test_quadcopter(;verbosity::Int = 0)
    nx = 6
    nu = 4                              # [ux,uy,uz,Γ]
    N = 30                              # number of nodes
    nh = 2 * N                          # two obstacles, enforced at each node

    # -------------------- setup problem -------------------- #
    prob, x_ref, u_ref = SCPLib.get_quadcopter_problem(Clarabel.Optimizer, N)

    @test prob.nx == nx
    @test prob.nu == nu
    @test prob.N == N
    @test prob.nh == nh
    @test size(x_ref) == (nx, N)
    @test size(u_ref) == (nu, N-1)

    # -------------------- instantiate algorithm -------------------- #
    algo = SCPLib.SCvxStar(nx, N; nh=nh, w0 = 10.0, l1_penalty = true)

    # solve problem
    solution = SCPLib.solve!(algo, prob, x_ref, u_ref;
        verbosity = verbosity, tol_opt = 1e-6, tol_feas = 1e-6)

    # propagate solution
    sols_opt, g_dynamics_opt = SCPLib.get_trajectory(prob, solution.x, solution.u)
    @test maximum(abs.(g_dynamics_opt)) <= 1e-6
    @test solution.status == :Optimal
    @test solution.info[:J0][end] ≈ 89.54904 atol=1e-3

    # obstacles are avoided
    h_opt = prob.h_noncvx(prob.lincache, solution.x, solution.u)
    @test maximum(h_opt) <= 1e-6
end


test_quadcopter(verbosity = verbosity)
