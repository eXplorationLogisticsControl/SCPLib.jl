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
Scale trust-region radii. `NaN` entries are copied through and are not used as radii.
Mixed problems mark the unused continuous column `N` this way.
"""
function scale_trust_radii(Δ::AbstractMatrix, factor::Real, bound::Real, shrink::Bool)
    out = similar(Δ)
    for i in eachindex(Δ)
        v = Δ[i]
        if isnan(v)
            out[i] = v
        elseif shrink
            out[i] = max(v / factor, bound)
        else
            out[i] = min(v * factor, bound)
        end
    end
    return out
end


function radii_at_lower_bound(Δ::AbstractMatrix, Δ_min::Real)
    saw_finite = false
    for v in Δ
        if isnan(v)
            continue
        end
        saw_finite = true
        if v != Δ_min
            return false
        end
    end
    return saw_finite
end


"""
    update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)

Update trust-region size Δ.
Returns a flag indicating whether the trust-region size was updated.
`NaN` radii are left unchanged.
"""
function update_trust_region!(algo::SCPAlgorithm, rho_i::Float64)
    flag_trust_region = false
    if rho_i < algo.rhos[2]
        algo.tr.Δ = scale_trust_radii(algo.tr.Δ, algo.alphas[1], algo.Δ_bounds[1], true)
        flag_trust_region = true
    elseif rho_i >= algo.rhos[3]
        algo.tr.Δ = scale_trust_radii(algo.tr.Δ, algo.alphas[2], algo.Δ_bounds[2], false)
        flag_trust_region = true
    end

    if algo.use_trustregion_control
        if rho_i < algo.rhos[2]
            algo.tr_u.Δ = scale_trust_radii(algo.tr_u.Δ, algo.alphas[1], algo.Δ_bounds[1], true)
            flag_trust_region = true
        elseif rho_i >= algo.rhos[3]
            algo.tr_u.Δ = scale_trust_radii(algo.tr_u.Δ, algo.alphas[2], algo.Δ_bounds[2], false)
            flag_trust_region = true
        end
    end
    return flag_trust_region
end


"""
    trust_region_at_lower_bound(algo::SCPAlgorithm)

True when every trust-region radius is already at its lower bound, so a rejected
step cannot shrink the subproblem any further.
"""
function trust_region_at_lower_bound(algo::SCPAlgorithm)
    Δ_min = algo.Δ_bounds[1]
    at_bound = radii_at_lower_bound(algo.tr.Δ, Δ_min)
    if algo.use_trustregion_control
        at_bound &= radii_at_lower_bound(algo.tr_u.Δ, Δ_min)
    end
    return at_bound
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
    @constraint(prob.model, constraint_trust_region_x_lb[k in 1:size(x_ref,2)],
        -(prob.model[:x][:,k] - x_ref[:,k]) <= algo.tr.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_x_ub[k in 1:size(x_ref,2)],
          prob.model[:x][:,k] - x_ref[:,k]  <= algo.tr.Δ[:,k])

    if algo.use_trustregion_control
        add_control_trust_region!(algo, prob, u_ref)
    end
    return
end


function add_control_trust_region!(algo::SCPAlgorithm, prob::OptimalControlProblem, u_ref)
    @constraint(prob.model, constraint_trust_region_u_lb[k in 1:size(u_ref,2)],
        -(prob.model[:u][:,k] - u_ref[:,k]) <= algo.tr_u.Δ[:,k])
    @constraint(prob.model, constraint_trust_region_u_ub[k in 1:size(u_ref,2)],
        prob.model[:u][:,k] - u_ref[:,k]  <= algo.tr_u.Δ[:,k])
    return
end
