"""Dynamics utilities"""


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
"""
function get_continuous_augmented_eom(eom!::Function, f_dfdx::Function, f_dfdu::Function, nx::Int, nu::Int)
    eom_aug! = function (dx_aug, x_aug, p, t)
        eom!(dx_aug, x_aug, p, t)

        A = f_dfdx(x_aug[1:nx], p.u, p, t)
        B = f_dfdu(x_aug[1:nx], p.u, p, t)
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx))', nx*nx)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
        return
    end
    return eom_aug!
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
"""
function get_continuous_augmented_eom(eom!::Function, params, nx::Int)
    @assert hasfield(typeof(params), :u) "params must have field 'u'"

    nu = length(params.u)
    p_dual = deepcopy(params)

    f_u2dx! = function (dx,x,u,t)
        p_dual.u = u             # replace control with dual vector
        eom!(dx,x,p_dual,t)
    end

    placeholder = zeros(nx)
 
    eom_aug! = function (dx_aug, x_aug, p, t)
        eom!(dx_aug, x_aug, p, t)

        A = ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t), placeholder, x_aug[1:nx])
        B = ForwardDiff.jacobian((y,u) -> f_u2dx!(y,x_aug,u,t), dx_aug[1:nx], p.u[:])

        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx))', nx*nx)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
        return
    end
    return eom_aug!
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
"""
function get_impulsive_augmented_eom(eom!::Function, f_dfdx::Function, nx::Int)
    eom_aug! = function (dx_aug, x_aug, p, t)
        eom!(dx_aug, x_aug, p, t)
        A = f_dfdx(x_aug[1:nx], p.u, p, t)
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx))', nx*nx)
        return
    end
    return eom_aug!
end


"""
Get augmented in-place dynamics function with signature `eom_aug!(dx_aug, x_aug, p, t)`
"""
function get_impulsive_augmented_eom(eom!::Function, params, nx::Int)
    @assert hasfield(typeof(params), :u) "params must have field 'u'"
    placeholder = zeros(nx)
 
    eom_aug! = function (dx_aug, x_aug, p, t)
        eom!(dx_aug, x_aug, p, t)
        A = ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t), placeholder, x_aug[1:nx])
        dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx))', nx*nx)
        return
    end
    return eom_aug!
end
