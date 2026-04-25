"""MWE for state-transition tensor"""

using LinearAlgebra
using ForwardDiff

p = [1.0]

function eom!(dx, x, p, t)
    dx[1:3] = x[4:6]
    dx[4:6] = -p[1] / norm(x[1:3])^3 * x[1:3]
end

t0 = 0.0
x0 = [1.0, 0.1, 0.2, 0.3, 1.0, -0.1]
nx = length(x0)

# Jacobian (works)
A1 = ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t0), zeros(nx), x0)

# Hessian (doesn't work)
A2 = ForwardDiff.jacobian(x -> ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t0), zeros(nx), x), x0)
A2 = reshape(A2, nx, nx, nx)