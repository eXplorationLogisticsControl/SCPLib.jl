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
