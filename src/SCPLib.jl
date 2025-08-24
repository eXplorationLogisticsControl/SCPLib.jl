module SCPLib

using JuMP
using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Printf
using SparseDiffTools

abstract type OptimalControlProblem end
abstract type SCPAlgorithm end

include("utils.jl")
include("dynamics.jl")

include("linearization.jl")
include("continuous_problem.jl")
include("impulsive_problem.jl")
include("constraints.jl")
include("trust_region.jl")

include("algorithms/scvxstar.jl")
include("algorithms/fixedtrw.jl")
include("algorithms/proxlinear.jl")

export get_continuous_augmented_eom

export ContinuousProblem
export get_trajectory, get_trajectory_augmented

export SCvxStarSolution, solve_convex_subproblem!, solve!

end # module SCPLib
