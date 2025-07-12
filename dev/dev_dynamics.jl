"""Quadcoptor example with SCvx*"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


# -------------------- setup problem -------------------- #
# system parameters
nx = 6
nu = 4                              # [ux,uy,uz,Γ]
g = [-9.81, 0, 0]
k_D = 0.5
t_N = 5;                        # s, duration of problem
m = 0.3;                        # kg, mass of quadrotor
T_min = 1.0;                    # N, minimum thrust
T_max = 4.0;                    # N, maximum thrust
theta_max = pi/4;               # rad, maximum tilt angle
N = 30;                         # number of nodes

# initial and final states
x_initial = [0, 0, 0, 0, 0.5, 0];
x_final = [0, 10, 0, 0, 0.5, 0];

# obstacle avoidance parameters
R_obstacle_1 = 1.0              # m, radius of obstacle 1
p_obstacle_1 = [0, 3, 0.45]     # m, position of obstacle 1
R_obstacle_2 = 1.0              # m, radius of obstacle 2
p_obstacle_2 = [0, 7, -0.45]    # m, position of obstacle 2

# ODE parameters
mutable struct QuadroptorParams
    u::Vector
end

params = QuadroptorParams(zeros(nu))

# rhs and jacobian expressions for quadrotor dynamics
function quadrotor_dfdx(x, u, p, t)
    v = x[4:6]
    v_norm = norm(v)
    dfdx = [zeros(3,3) I(3);
            zeros(3,3)  (-k_D * (v_norm * I(3) + (v * v') / v_norm))]
    return dfdx
end

function quadrotor_dfdu(x, u, p, t)
    dfdu = [zeros(3,4); 1/m * I(3) zeros(3,1)];
    return dfdu
end

function quadrotor_rhs!(dx, x, p, t)
    dx[1:3] = x[4:6]
    dx[4:6] = -k_D*norm(x[4:6])*x[4:6] + g
    B = quadrotor_dfdu(x[1:6], p.u, p, t)
    dx[1:6] += B * p.u
    return
end

# eom_aug! = SCPLib.get_continuous_augmented_eom(quadrotor_rhs!, quadrotor_dfdx, quadrotor_dfdu, nx, nu)
eom_aug! = SCPLib.get_continuous_augmented_eom(quadrotor_rhs!, nx, nu)


f_u2dx! = function (dx,x,u,p,t)
    #p_copy = deepcopy(p)
    #p_copy.u = u
    #quadrotor_rhs!(dx,x,p_copy,t)
    
    p.u = u
    quadrotor_rhs!(dx,x,p,t)
end

@show params
u_try = [0.1,0.2,0.3,sqrt(0.1^2 + 0.2^2 + 0.3^2)]
dx_foo = zeros(nx)
f_u2dx!(dx_foo, x_initial, u_try, params, 0.0)

@show params
@show dx_foo

# analytically compute B
B = quadrotor_dfdu(x_initial, u_try, params, 0.0)

# automatic differentiaiton B
B_ad = ForwardDiff.jacobian((y,u) -> f_u2dx!(y,x_initial,u,params,0.0), dx_foo, u_try)
#B_ad = ForwardDiff.jacobian((y,u) -> f_u2dx!(y,x_initial,u,params,0.0), dx_foo[:], u_try[:])

# numerically compute B
B_num = zeros(nx,nu)
for i in 1:nu
    u_copy = deepcopy(u_try)
    u_copy[i] += 1e-6
    dx_foo_copy = zeros(nx)
    f_u2dx!(dx_foo_copy, x_initial, u_copy, params, 0.0)
    B_num[:,i] = (dx_foo_copy - dx_foo) / 1e-6
end

@show B_num