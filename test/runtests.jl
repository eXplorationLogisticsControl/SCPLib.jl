"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

# test setting
verbosity = 0
get_plot = false

@testset "SCvxStar" begin
    include("test_scvxstar_subproblem.jl")
    include("test_scvxstar_dynamics_userdefined.jl")
    include("test_scvxstar_dynamics_ad.jl")
    include("test_scvxstar_h_noncvx.jl")

    include("test_scvxstar_impulsive_dynamics_only.jl")
    include("test_scvxstar_custom_propagate_func.jl")
end

@testset "Biased control" begin
    include("test_scvxstar_impulsive_u_bias.jl")
    include("test_scvxstar_continuous_u_bias.jl")
end

@testset "ProxLinear" begin
    include("test_proxlinear_dynamics_only.jl")
end

@testset "SCvx" begin
    include("test_scvx_dynamics_ad.jl")
end