"""Prox-linear algorithm"""


mutable struct ProxLinear <: SCPAlgorithm
end


function Base.show(io::IO, algo::ProxLinear)
    println(io, "Prox-linear algorithm")
end


function solve!(
    algo::ProxLinear,
    prob::OptimalControlProblem,
    x_ref, u_ref, y_ref;
    maxiter::Int = 1,
)
end