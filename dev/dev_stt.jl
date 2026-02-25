"""Develop state-transition tensor"""

using Clarabel
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParams_dynamics_ad
    μ::Float64
    u::Vector
    function ControlParams_dynamics_ad(μ::Float64)
        new(μ, zeros(4))
    end
end


μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU/TU      # km/s
params = ControlParams_dynamics_ad(μ)

function eom!(drv, rv, p, t)
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
    r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
    drv[1:3] = rv[4:6]
    # derivatives of velocities
    drv[4] =  2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x);
    drv[5] = -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y;
    drv[6] = -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z;
    # append controls
    drv[4:6] += p.u[1:3]
    return
end


function eom(rv, p, t)
    x, y, z = rv[1:3]
    vx, vy, vz = rv[4:6]
    r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
    r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
    drv = [
        rv[4:6];
        2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x);
        -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y;
        -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z;
    ]
    # append controls
    drv[4:6] += p.u[1:3]
    return drv
end


rv0 = [1.0809931218390707E+00,
    0.0000000000000000E+00,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15]
period_0 = 2.3538670417546639E+00


# initial & final LPO
sol_lpo0 = solve(
    ODEProblem(eom!, rv0, [0.0, period_0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
 
eom_stm! = function (dx_aug, x_aug, p, t)
    eom!(view(dx_aug,1:6), x_aug[1:6], p, t)
    A = ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t), zeros(nx), x_aug[1:nx])
    dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx)')', nx*nx)
    return
end

eom_stm_stt! = function (dx_aug, x_aug, p, t)
    eom!(view(dx_aug,1:6), x_aug[1:6], p, t)
    A = ForwardDiff.jacobian((y,x) -> eom!(y,x,p,t), zeros(nx), x_aug[1:nx])
    dx_aug[nx+1:nx*(nx+1)] = reshape((A * reshape(x_aug[nx+1:nx*(nx+1)],nx,nx)')', nx*nx)
    return
end

# from:
# https://juliadiff.org/ForwardDiff.jl/stable/user/advanced/#Hessian-of-a-vector-valued-function
function vector_hessian(f, x)
    n = length(x)
    A = ForwardDiff.jacobian(f, x)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return A, reshape(out, n, n, n)
end


function vector_hessian_inplace(f, y, x)
    n = length(x)
    A = ForwardDiff.jacobian((y,x) -> f(y,x), y, x)
    out = 0
    # out = ForwardDiff.jacobian((y,x) -> ForwardDiff.jacobian(x -> f(y,x), x), y, x)
    #out = ForwardDiff.jacobian((y,x) -> ForwardDiff.jacobian((y,x) -> f(y,x), y, x), y, x)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian((_y,x) -> f(_y,x), zeros(nx), x), x)
    # println("Hessian computation")
    return A, out #reshape(out, n, n, n)
end


# stm via finite difference
stm_fd = zeros(6,6)
h = 1e-6
for i in 1:6
    x_plus = deepcopy(rv0)
    x_plus[i] += h
    _sol = solve(ODEProblem(eom!, x_plus, [0.0, period_0], params), Tsit5(); reltol = 1e-12, abstol = 1e-12)
    stm_fd[:,i] = (_sol.u[end][1:6] - sol_lpo0.u[end][1:6]) / h
end

# initial & final LPO
sol_stm = solve(
    ODEProblem(eom_stm!, [rv0; vec(I(6))], [0.0, period_0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
stm_f = reshape(sol_stm.u[end][nx+1:nx*(nx+1)], (nx,nx))'

# Jac & Hessian with explicit eom function
A1, A2 = vector_hessian(x -> eom(x,params,0.0), rv0)

# hand assembled
A2_hand = zeros(6,6,6)
for i in 1:6
    A2_hand[i,:,:] = ForwardDiff.hessian(x -> eom(x,params,0.0)[i], rv0)
end
@assert A2 == A2_hand

#ForwardDiff.hessian(x -> eom(x,params,0.0)[1], rv0)
#ForwardDiff.hessian((y,x) -> eom!(y,x,params,0.0)[1], zeros(6), rv0)

# Jac & Hessian with implicit eom function
A1imp, A2imp = vector_hessian_inplace((y,x) -> eom!(y,x,params,0.0), zeros(6), rv0)


println("Done!")