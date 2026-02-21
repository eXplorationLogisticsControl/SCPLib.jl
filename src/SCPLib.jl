module SCPLib

using JuMP
using LinearAlgebra
using ForwardDiff
using OrdinaryDiffEq
using Printf
using SparseDiffTools

abstract type OptimalControlProblem end
abstract type SCPAlgorithm end
abstract type SCPSolution end

abstract type TrustRegionAlgorithm <: SCPAlgorithm end

include("utils.jl")
include("dynamics.jl")

include("linearization.jl")
include("continuous_problem.jl")
include("impulsive_problem.jl")
include("nlconstraints.jl")
include("trust_region.jl")

include("algorithms/scvx.jl")
include("algorithms/scvxstar.jl")
include("algorithms/fixedtrw.jl")
include("algorithms/proxlinear.jl")

include("problems/quadcopter.jl")

export get_continuous_augmented_eom
export get_impulsive_augmented_eom

export ContinuousProblem
export ImpulsiveProblem
export get_trajectory, get_trajectory_augmented
export stack_flatten_variables, unpack_flattened_variables

export SCvx, SCvxSolution
export SCvxStar, SCvxStarSolution
export FixedTRWSCP, FixedTRWSCPSolution
export ProxLinear, ProxLinearSolution
export solve_convex_subproblem!, solve!

end # module SCPLib
