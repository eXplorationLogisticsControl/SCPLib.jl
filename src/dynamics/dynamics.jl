"""Dynamics utilities"""


"""Bundle immutable physics params with segment control for ODEProblem."""
dynamics_input(params, u::AbstractVector) = (; params, u)

"""Preallocate one control buffer per shooting segment (thread-safe with EnsembleThreads)."""
make_u_pool(nu::Int, n_segments::Int) = [zeros(nu) for _ in 1:n_segments]

function fill_segment_control!(u_k, u_ref, u_bias, k)
    u_k .= @view(u_ref[:, k]) .+ @view(u_bias[:, k])
    return u_k
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
where `p = (; params, u)`.
"""
function get_continuous_augmented_eom(eom!::Function, f_dfdx::Function, f_dfdu::Function, nx::Int, nu::Int)
    eom_aug! = function (dx_aug, x_aug, p, t)
        params, u = p.params, p.u
        x = x_aug[1:nx]
        eom!(view(dx_aug, 1:nx), x, p, t)

        A = f_dfdx(x, u, params, t)
        B = f_dfdu(x, u, params, t)
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)], nx, nx)), nx^2)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nx, nu)) + B), nx*nu)
        return
    end
    return eom_aug!
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
where `p = (; params, u)`.
"""
function get_continuous_augmented_eom(eom!::Function, params, nx::Int, nu::Int)
    eom_aug! = function (dx_aug, x_aug, p, t)
        params_seg, u = p.params, p.u
        x = x_aug[1:nx]
        eom!(view(dx_aug, 1:nx), x, p, t)

        A = ForwardDiff.jacobian((y, x_) -> eom!(y, x_, (; params=params_seg, u=u), t), zeros(nx), x)
        B = ForwardDiff.jacobian((y, u_) -> eom!(y, x, (; params=params_seg, u=u_), t), zeros(nx), u)

        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)], nx, nx)), nx^2)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nx, nu)) + B), nx*nu)
        return
    end
    return eom_aug!
end


"""Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`"""
function get_impulsive_augmented_eom(eom!::Function, f_dfdx::Function, nx::Int)
    eom_aug! = function (dx_aug, x_aug, p, t)
        params, u = p.params, p.u
        eom!(dx_aug, x_aug, p, t)
        A = f_dfdx(x_aug[1:nx], u, params, t)
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)], nx, nx)), nx^2)
        return
    end
    return eom_aug!
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
where `p = (; params, u)`.
"""
function get_impulsive_augmented_eom(eom!::Function, params, nx::Int)
    placeholder = zeros(nx)

    eom_aug! = function (dx_aug, x_aug, p, t)
        eom!(dx_aug, x_aug, p, t)
        A = ForwardDiff.jacobian((y, x) -> eom!(y, x, p, t), placeholder, x_aug[1:nx])
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)], nx, nx)), nx^2)
        return
    end
    return eom_aug!
end
