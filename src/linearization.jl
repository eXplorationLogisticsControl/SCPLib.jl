"""Handling continuous dynamics"""


"""Cache for linearized Jacobians"""
mutable struct LinearizedCache
    Φ_A::Array{Float64, 3}
    Φ_B::Array{Float64, 3}
    Φ_c::Array{Float64, 2}

    ∇g::Array{Float64, 2}
    ∇h::Array{Float64, 2}

    function LinearizedCache(nx::Int, nu::Int, N::Int, ng::Int, nh::Int)
        return new(
            zeros(nx, nx, N-1),
            zeros(nx, nu, N-1),
            zeros(nx, N-1),
            zeros(ng, nx*N + nu*(N-1)),
            zeros(nh, nx*N + nu*(N-1)),
        )
    end
end


"""Set cache for dynamics"""
function set_dynamics_cache!(lincache::LinearizedCache, x_ref, u_ref, sols::Union{Vector{ODESolution},EnsembleSolution})
    nx, N = size(x_ref)
    nu, _ = size(u_ref)
    for k in 1:N-1
        xf_aug = sols[:,k].u[end]
        lincache.Φ_A[:,:,k] = reshape(xf_aug[nx+1:nx*(nx+1)], (nx,nx))'
        lincache.Φ_B[:,:,k] = reshape(xf_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))'
        lincache.Φ_c[:,k]   = xf_aug[1:nx] - lincache.Φ_A[:,:,k] * x_ref[:,k] - lincache.Φ_B[:,:,k] * u_ref[:,k]
    end
end


function set_g_noncvx_cache!(lincache::LinearizedCache, ∇g_noncvx::Function, x_ref, u_ref, y_ref)
    lincache.∇g[:,:] = ∇g_noncvx(x_ref, u_ref, y_ref)
    return
end


function set_h_noncvx_cache!(lincache::LinearizedCache, ∇h_noncvx::Function, x_ref, u_ref, y_ref)
    lincache.∇h[:,:] = ∇h_noncvx(x_ref, u_ref, y_ref)
    return
end