"""Handling continuous dynamics"""


"""Cache for linearized Jacobians"""
mutable struct MultipleShootingCache <: AbstractLinearizationCache
    Φ_A::Array{Float64, 3}
    Φ_B::Array{Float64, 3}
    Φ_c::Array{Float64, 2}

    ∇g::Array{Float64, 2}
    ∇h::Array{Float64, 2}

    function MultipleShootingCache(nx::Int, nu::Int, N::Int, Nu::Int, ng::Int, nh::Int)
        return new(
            zeros(nx, nx, N-1),
            zeros(nx, nu, N-1),
            zeros(nx, N-1),
            zeros(ng, nx*N + nu*Nu),
            zeros(nh, nx*N + nu*Nu),
        )
    end
end


"""Set cache for continuous dynamics"""
function set_continuous_dynamics_cache!(lincache::MultipleShootingCache, x_ref, u_ref, sols::Union{Vector{ODESolution},EnsembleSolution})
    nx, N = size(x_ref)
    nu, _ = size(u_ref)
    for (k,sol) in enumerate(sols)
        xf_aug = sol.u[end]
        lincache.Φ_A[:,:,k] = reshape(xf_aug[nx+1:nx*(nx+1)], (nx,nx))
        lincache.Φ_B[:,:,k] = reshape(xf_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nx,nu))
        lincache.Φ_c[:,k]   = xf_aug[1:nx] - lincache.Φ_A[:,:,k] * x_ref[:,k] - lincache.Φ_B[:,:,k] * u_ref[:,k]
    end
end


"""Set cache for impulsive dynamics"""
function set_impulsive_dynamics_cache!(lincache::MultipleShootingCache, x_ref, u_ref, sols::Union{Vector{ODESolution},EnsembleSolution}, dfdu::Function)
    nx, N = size(x_ref)
    for (k,sol) in enumerate(sols)
        xf_aug = sol.u[end]
        lincache.Φ_A[:,:,k] = reshape(xf_aug[nx+1:nx*(nx+1)], (nx,nx))
        lincache.Φ_B[:,:,k] = lincache.Φ_A[:,:,k] * dfdu(x_ref[:,k], u_ref[:,k], sol.t[1])
        lincache.Φ_c[:,k]   = xf_aug[1:nx] - lincache.Φ_A[:,:,k] * x_ref[:,k] - lincache.Φ_B[:,:,k] * u_ref[:,k]
    end
end


"""Set cache for non-convex equality constraints"""
function set_g_noncvx_cache!(lincache::MultipleShootingCache, ∇g_noncvx::Function, x_ref, u_ref)
    lincache.∇g[:,:] = ∇g_noncvx(x_ref, u_ref)
    return
end


"""Set cache for non-convex inequality constraints"""
function set_h_noncvx_cache!(lincache::MultipleShootingCache, ∇h_noncvx::Function, x_ref, u_ref)
    lincache.∇h[:,:] = ∇h_noncvx(x_ref, u_ref)
    return
end


mutable struct ForwardBackwardCache <: AbstractLinearizationCache
    ∇g_dyn::Array{Float64, 2}
    ∇g::Array{Float64, 2}
    ∇h::Array{Float64, 2}

    function ForwardBackwardCache(nx::Int, nu::Int, N::Int, Nu::Int, ng::Int, nh::Int)
        return new(
            zeros(nx, nx*2 + nu*Nu),
            zeros(ng, nx*2 + nu*Nu),
            zeros(nh, nx*2 + nu*Nu),
        )
    end
end


"""Set cache for continuous dynamics"""
function set_continuous_dynamics_cache!(lincache::ForwardBackwardCache, x_ref, u_ref, sols::Vector{ODESolution})
    nx, _ = size(x_ref)
    nu, Nu = size(u_ref)
    Nu_fwd = div(Nu+1, 2)

    Φ_A_list = [reshape(sols[k].u[end][nx+1:nx*(nx+1)], (nx,nx))' for k in 1:Nu]
    Φ_B_list = [reshape(sols[k].u[end][nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' for k in 1:Nu]

    lincache.∇g_dyn[1:nx,1:nx] = prod(reverse(Φ_A_list[1:Nu_fwd]))
    lincache.∇g_dyn[1:nx,nx+1:2nx] = prod(Φ_A_list[Nu_fwd+1:end])

    for k in 1:Nu_fwd
        if k < Nu_fwd 
            lincache.∇g_dyn[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = -prod(reverse(Φ_A_list[k+1:Nu_fwd])) * Φ_B_list[k]
        else
            lincache.∇g_dyn[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = -Φ_B_list[k]
        end
    end

    for k in Nu_fwd+1:Nu
        if k > Nu_fwd+1
            lincache.∇g_dyn[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = prod(Φ_A_list[Nu_fwd+1:k-1]) * Φ_B_list[k]
        else
            lincache.∇g_dyn[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = Φ_B_list[k]
        end
    end
    return Φ_A_list, Φ_B_list
end