"""Run tests"""

using Test

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

# @testset "AD dynamics" begin
#     include("test_ad_dynamics.jl")
# end

get_plot = false

@testset "SCvxStar" begin
    include("test_scvxstar_subproblem.jl")
    include("test_scvxstar_dynamics_userdefined.jl")
    include("test_scvxstar_h_noncvx.jl")

    include("test_scvxstar_impulsive_dynamics_only.jl")
end
