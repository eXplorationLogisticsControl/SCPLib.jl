"""Functions & structs for trust-region handling"""

"""Trust-region struct"""
mutable struct TrustRegions
    Δ::Matrix{Float64}
end


"""Initialize trust-region struct from a single scalar"""
function TrustRegions(nx::Int, N::Int, Δ0::Float64)
    return TrustRegions(Δ0 * ones(nx, N))
end


"""Initialize trust-region struct from a vector of length `nx`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Vector{Float64})
    @assert length(Δ0s) == nx "Length of Δ0s must match number of states"
    return TrustRegions(repeat(Δ0s, 1, N))
end


"""Initialize trust-region struct from a matrix of size `(nx, N)`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Matrix{Float64})
    @assert size(Δ0s) == (nx, N) "Size of Δ0s must match number of states and number of nodes"
    return TrustRegions(Δ0s)
end


"""Update trust-region size"""
function update_trust_region!(algo::TrustRegionAlgorithm, rho_i::Float64)
    flag_trust_region = false
    if rho_i < algo.rhos[2]
        algo.tr.Δ = max.(algo.tr.Δ / algo.alphas[1], algo.Δ_bounds[1])
        flag_trust_region = true
    elseif rho_i >= algo.rhos[3]
        algo.tr.Δ = min.(algo.tr.Δ * algo.alphas[2], algo.Δ_bounds[2])
        flag_trust_region = true
    end
    return flag_trust_region
end


"""Set trust-region constraints"""
function set_trust_region_constraints!(algo::TrustRegionAlgorithm, prob::OptimalControlProblem, x_ref::Union{Matrix,Adjoint}, u_ref::Union{Matrix,Adjoint})
    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.tr.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.tr.Δ[:,k])
    return
end
