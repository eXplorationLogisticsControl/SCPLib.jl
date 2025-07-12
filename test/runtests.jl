"""Run tests"""


using Test

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

@testset "AD dynamics" begin
    include("test_ad_dynamics.jl")
end

@testset "SCvxStar" begin
    include("test_convex_subproblem.jl")
end
