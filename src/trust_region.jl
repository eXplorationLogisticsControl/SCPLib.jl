"""Functions & structs for trust-region handling"""

"""Trust-region struct"""
mutable struct TrustRegions
    Δ::Matrix{Float64}
    Δ_type::Symbol
end


function Base.show(io::IO, tr::TrustRegions)
    println(io, "Trust-region struct")
    @printf("   Trust-region type   : %s\n", tr.Δ_type)
    @printf("   Trust-region size Δ : %1.2e\n", tr.Δ[1,1])
end


"""Initialize trust-region struct from a single scalar"""
function TrustRegions(nx::Int, N::Int, Δ0::Float64)
    return TrustRegions(Δ0 * ones(nx, N), :Uniform)
end


"""Initialize trust-region struct from a vector of length `nx`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Vector{Float64})
    @assert length(Δ0s) == nx "Length of Δ0s must match number of states"
    return TrustRegions(repeat(Δ0s, 1, N), :NodeDependent)
end


"""Initialize trust-region struct from a matrix of size `(nx, N)`"""
function TrustRegions(nx::Int, N::Int, Δ0s::Matrix{Float64}, x_dependent::Bool = false)
    @assert size(Δ0s) == (nx, N) "Size of Δ0s must match number of states and number of nodes"
    if x_dependent
        Δ_type = :StateDependent
    else
        Δ_type = :Independent
    end
    return TrustRegions(Δ0s, Δ_type)
end



"""
    update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)

Update trust-region size Δ
Returns a flag indicating whether the trust-region size was updated
"""
function update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)
    flag_trust_region = false
    if rho_i < algo.rhos[2]
        algo.tr.Δ = max.(algo.tr.Δ / algo.alphas[1], algo.Δ_bounds[1])
        flag_trust_region = true
    elseif rho_i >= algo.rhos[3]
        algo.tr.Δ = min.(algo.tr.Δ * algo.alphas[2], algo.Δ_bounds[2])
        flag_trust_region = true
    end

    if algo.use_trustregion_control
        if rho_i < algo.rhos[2]
            algo.tr_u.Δ = max.(algo.tr_u.Δ / algo.alphas[1], algo.Δ_bounds[1])
            flag_trust_region = true
        elseif rho_i >= algo.rhos[3]
            algo.tr_u.Δ = min.(algo.tr_u.Δ * algo.alphas[2], algo.Δ_bounds[2])
            flag_trust_region = true
        end
    end
    return flag_trust_region
end


"""
    set_trust_region_constraints!(
        algo::SCPAlgorithm,
        prob::OptimalControlProblem,
        x_ref::Union{Matrix,Adjoint},
        u_ref::Union{Matrix,Adjoint},
    )
    
Set trust-region constraints to JuMP model in `prob`
"""
function set_trust_region_constraints!(
    algo::SCPAlgorithm,
    prob::OptimalControlProblem,
    x_ref::Union{Matrix,Adjoint},
    u_ref::Union{Matrix,Adjoint},
)
    # define trust-region constraints
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:prob.N],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.tr.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:prob.N],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.tr.Δ[:,k])

    if algo.use_trustregion_control
        @constraint(prob.model, constraint_trust_region_u_lb[k in 1:size(u_ref,2)],
            -(prob.model[:u][:,k] - u_ref[:,k]) <= algo.tr_u.Δ[:,k])
        @constraint(prob.model, constraint_trust_region_u_ub[k in 1:prob.N],
            prob.model[:u][:,k] - u_ref[:,k]  <= algo.tr_u.Δ[:,k])
    end
    return
end