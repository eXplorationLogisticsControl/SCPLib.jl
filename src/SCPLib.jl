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

include("linearization.jl")
include("continuous_problem.jl")

include("scvxstar.jl")
include("proxlinear.jl")

end # module SCPLib
