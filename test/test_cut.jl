"""Test conjugate unscented transform"""

using LinearAlgebra
using Random
using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

function _weighted_moments(X, w)
    mu = X * w
    Δ = X .- mu
    Sigma = Δ * Diagonal(w) * Δ'
    return mu, Sigma
end

function _mc_moments(samples)
    N = size(samples, 2)
    mu = vec(sum(samples, dims=2)) / N
    Δ = samples .- mu
    Sigma = (Δ * Δ') / N
    return mu, Sigma
end

function polar_to_cartesian(x)
    r, θ = x[1], x[2]
    return [r * cos(θ), r * sin(θ)]
end

"""E[r cos θ], E[r sin θ] for independent Gaussians r, θ."""
function polar_to_cartesian_mean(mu, σθ)
    scale = mu[1] * exp(-σθ^2 / 2)
    return [scale * cos(mu[2]), scale * sin(mu[2])]
end

function test_cut()
    @testset "sigma-point generators" begin
        expected_N = Dict(
            (2, 4) => 9,
            (3, 4) => 14,
            (6, 4) => 76,
            (2, 6) => 13,
            (4, 6) => 49,
            (5, 6) => 83,
            (6, 6) => 137,
            (7, 6) => 423,
            (2, 8) => 21,
            (3, 8) => 59,
            (4, 8) => 161,
            (5, 8) => 355,
            (6, 8) => 745,
        )
        for ((n, degree), Nexp) in expected_N
            X, w = SCPLib.cut_sigma_points(n, degree)
            @test size(X) == (n, Nexp)
            @test length(w) == Nexp
            @test sum(w) ≈ 1 atol=1e-12
            @test all(>=(0), w)
            mu, Sigma = _weighted_moments(X, w)
            @test mu ≈ zeros(n) atol=1e-12
            @test Sigma ≈ Matrix{Float64}(I, n, n) atol=1e-7
        end

        # Fourth-order moments for a 3D standard normal (CUT4+).
        X, w = SCPLib.cut_sigma_points(3, 4)
        m4 = sum(w[k] * X[1, k]^4 for k in axes(X, 2))
        m22 = sum(w[k] * X[1, k]^2 * X[2, k]^2 for k in axes(X, 2))
        @test m4 ≈ 3 atol=1e-10
        @test m22 ≈ 1 atol=1e-10

        @test_throws ArgumentError SCPLib.cut_sigma_points(1, 4)
        @test_throws ArgumentError SCPLib.cut_sigma_points(3, 5)
        @test_throws ArgumentError SCPLib.cut_sigma_points(10, 6)
        @test_throws ArgumentError SCPLib.cut_sigma_points(7, 8)
    end

    @testset "identity and affine maps" begin
        mu = [0.4, -1.2, 0.7]
        Sigma = [1.5 0.2 -0.1; 0.2 0.8 0.05; -0.1 0.05 1.1]
        Sigma = Hermitian(0.5 * (Sigma + Sigma'))
        A = [1.0 0.2 -0.3; 0.0 0.7 0.1; 0.4 -0.2 1.3]
        b = [0.1, -0.5, 0.2]

        for degree in (4, 6, 8)
            mu_id, Sigma_id = SCPLib.cut(mu, Sigma, identity; degree=degree)
            @test mu_id ≈ mu atol=1e-10
            @test Sigma_id ≈ Matrix(Sigma) atol=1e-8

            mu_aff, Sigma_aff = SCPLib.cut(mu, Sigma, x -> A * x + b; degree=degree)
            @test mu_aff ≈ A * mu + b atol=1e-10
            @test Sigma_aff ≈ A * Sigma * A' atol=1e-8
        end
    end

    @testset "Monte Carlo polar-to-Cartesian" begin
        mu = [1.0, π / 2]
        σr = 0.02
        σθ = deg2rad(15)
        Sigma = Diagonal([σr^2, σθ^2])
        mu_analytical = polar_to_cartesian_mean(mu, σθ)

        rng = MersenneTwister(42)
        Nmc = 100_000
        L = Matrix(sqrt(Hermitian(Matrix(Sigma))))
        samples = L * randn(rng, 2, Nmc) .+ mu
        Zmc = reduce(hcat, polar_to_cartesian.(eachcol(samples)))
        _, Sigma_mc = _mc_moments(Zmc)

        for degree in (4, 6, 8)
            mu_cut, Sigma_cut = SCPLib.cut(mu, Sigma, polar_to_cartesian; degree=degree)
            @test mu_cut ≈ mu_analytical atol=1e-8
            @test Sigma_cut ≈ Sigma_mc rtol=0.05 atol=5e-4
        end

        # Non-symmetric heading so E[x] is not identically 0.
        mu_ns = [1.0, π / 4]
        mu_cut_ns, _ = SCPLib.cut(mu_ns, Sigma, polar_to_cartesian; degree=4)
        @test mu_cut_ns ≈ polar_to_cartesian_mean(mu_ns, σθ) atol=1e-6
    end

    @testset "Monte Carlo mixed-dimension map" begin
        mu = [0.3, -0.2, 0.5]
        Sigma = Diagonal([0.15, 0.2, 0.1])
        f(x) = [x[1]^2 + 0.5 * x[2], sin(x[3]) * exp(0.1 * x[2])]

        rng = MersenneTwister(7)
        Nmc = 200_000
        L = Matrix(sqrt(Hermitian(Matrix(Sigma))))
        samples = L * randn(rng, 3, Nmc) .+ mu
        Zmc = reduce(hcat, f.(eachcol(samples)))
        mu_mc, Sigma_mc = _mc_moments(Zmc)

        for degree in (4, 6, 8)
            mu_cut, Sigma_cut = SCPLib.cut(mu, Sigma, f; degree=degree)
            @test length(mu_cut) == 2
            @test size(Sigma_cut) == (2, 2)
            @test mu_cut ≈ mu_mc rtol=0.03 atol=5e-3
            @test Sigma_cut ≈ Sigma_mc rtol=0.08 atol=5e-3
        end
    end
end

test_cut()
