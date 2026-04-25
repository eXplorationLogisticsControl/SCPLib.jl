"""Forward-backward shooting gradient"""

using Clarabel
using ForwardDiff
using GLMakie
using JuMP
using LinearAlgebra
using OrdinaryDiffEq
using Random

include(joinpath(@__DIR__, "../src/SCPLib.jl"))

seed = 1234
Random.seed!(seed)

# -------------------- setup problem -------------------- #
# create parameters with `u` entry
mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end

μ = 1.215058560962404e-02
DU = 389703     # km
TU = 382981     # sec
MU = 500.0      # kg
VU = DU/TU      # km/s
params = ControlParams(μ)

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

function eom_aug!(dx_aug, x_aug, p, t)
    # state derivatives
    eom!(view(dx_aug, 1:6), x_aug[1:6], p, t)
    
    # STM derivatives
    r1vec = [x_aug[1] + p.μ, x_aug[2], x_aug[3]]
    r2vec = [x_aug[1] - 1 + p.μ, x_aug[2], x_aug[3]]
    G1 = (1 - params.μ) / norm(r1vec)^5*(3*r1vec*r1vec' - norm(r1vec)^2*I(3))
    G2 = params.μ / norm(r2vec)^5*(3*r2vec*r2vec' - norm(r2vec)^2*I(3))
    Omega = [0 2 0; -2 0 0; 0 0 0]
    A = [zeros(3,3)                  I(3);
         G1 + G2 + diagm([1,1,0])    Omega]
    B = [zeros(3,4); I(3) zeros(3,1)]

    # derivatives of Phi_A, Phi_B
    dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
    dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' + B)', nx*nu)
end

rv0 = [1.0809931218390707E+00,
    0.0000000000000000E+00,
    -2.0235953267405354E-01,
    1.0157158264396639E-14,
    -1.9895001215078018E-01,
    7.2218178975912707E-15]
period_0 = 2.3538670417546639E+00

rvf = [1.1648780946517576,
    0.0,
    -1.1145303634437023E-1,
    0.0,
    -2.0191923237095796E-1,
    0.0]
period_f = 3.3031221822879884

# -------------------- create problem -------------------- #
N = 6
nx = 6
nu = 4                              # [ux,uy,uz,Γ]
tf = 2.6 
times = LinRange(0.0, tf, N)

thrust = 0.35    # N
umax = thrust/MU/1e3 / (VU/TU)

# initial & final LPO
sol_lpo0 = solve(
    ODEProblem(eom!, rv0, [0.0, period_0], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)
sol_lpof = solve(
    ODEProblem(eom!, rvf, [0.0, period_f], params),
    Tsit5(); reltol = 1e-12, abstol = 1e-12
)

# create reference solution
x_along_lpo0 = sol_lpo0(LinRange(0.0, period_0, N))
x_along_lpof = sol_lpof(LinRange(0.0, period_f, N))
x_ref = zeros(nx,N)
alphas = LinRange(0,1,N)
for (i,alpha) in enumerate(alphas)
    x_ref[:,i] = (1-alpha)*x_along_lpo0[:,i] + alpha*x_along_lpof[:,i]
end

# control initial guess
u_ref = zeros(nu, N-1)
u_ref[1:3,:] = randn(3, N-1) * umax
u_ref[4,:] = norm.(eachcol(u_ref[1:3,:]))

# forward-backward shooting functions
function forward_shooting(x0, u_fwd, times, params; get_stm::Bool = false)
    _, Nu_fwd = size(u_fwd)
    xk = deepcopy(x0)
    sols = []
    for k in 1:Nu_fwd
        params.u[:] = u_fwd[:,k]
        if get_stm == false
            ode_problem = ODEProblem(eom!, xk, [times[k], times[k+1]], params)
        else
            ode_problem = ODEProblem(eom_aug!, SCPLib.init_continuous_dynamics_xaug(xk, nx, nu), [times[k], times[k+1]], params)
        end
        sol = solve(ode_problem, Tsit5(); reltol = 1e-12, abstol = 11e-12)
        xk = sol.u[end][1:nx]
        push!(sols, sol)
    end
    return xk, sols
end

# evaluate forward shooting
@show x_ref[:,1]
Nu_fwd = div(N,2)
u_fwd = u_ref[:, 1:Nu_fwd]
xmp_fwd, sols_fwd = forward_shooting(x_ref[:,1], u_fwd, times[1:Nu_fwd+1], params, get_stm=true)
@show x_ref[:,1]

# single shot
ode_problem = ODEProblem(eom_aug!, SCPLib.init_continuous_dynamics_xaug(x_ref[:,1], nx, nu), [times[1], times[Nu_fwd+1]], params)
sol_single_shot = solve(ode_problem, Tsit5(); reltol = 1e-12, abstol = 11e-12)
Φ_A_single_shot = reshape(sol_single_shot.u[end][nx+1:nx*(nx+1)], (nx,nx))'

# analytical jacobian
Φ_A_list = [reshape(sols_fwd[k].u[end][nx+1:nx*(nx+1)], (nx,nx))' for k in 1:Nu_fwd]
Φ_B_list = [reshape(sols_fwd[k].u[end][nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))' for k in 1:Nu_fwd]

jac_fwd_analytical = zeros(nx, 2nx + nu*Nu_fwd)
jac_fwd_analytical[1:6,1:6] = prod(reverse(Φ_A_list))
for k in 1:Nu_fwd
    if k < Nu_fwd 
        jac_fwd_analytical[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = prod(reverse(Φ_A_list[k+1:end])) * Φ_B_list[k]
    else
        jac_fwd_analytical[1:nx,2nx+1+nu*(k-1):2nx+k*nu] = Φ_B_list[k]
    end
end

# numerical jacobian
jac_fwd_finitediff = zeros(nx, 2nx + nu*Nu_fwd)
for i in 1:6
    x0_ptrb_pls = deepcopy(x_ref[:,1])
    x0_ptrb_pls[i] += 1e-6

    x0_ptrb_min = deepcopy(x_ref[:,1])
    x0_ptrb_min[i] -= 1e-6
    jac_fwd_finitediff[:,i] = (forward_shooting(x0_ptrb_pls, u_fwd, times[1:Nu_fwd+1], params)[1] - forward_shooting(x0_ptrb_min, u_fwd, times[1:Nu_fwd+1], params)[1]) / (2e-6)
end

for k in 1:Nu_fwd
    for i in 1:4
        u_fwd_pls = deepcopy(u_fwd)
        u_fwd_pls[i,k] += 1e-6
    
        u_fwd_min = deepcopy(u_fwd)
        u_fwd_min[i,k] -= 1e-6
        jac_fwd_finitediff[:,2nx+i+nu*(k-1)] = (forward_shooting(x_ref[:,1], u_fwd_pls, times[1:Nu_fwd+1], params)[1] - forward_shooting(x_ref[:,1], u_fwd_min, times[1:Nu_fwd+1], params)[1]) / (2e-6)
    end
end


# plot
fig = Figure(size=(1200,800))
ax3d = Axis3(fig[1,1]; aspect=:data)
for sol in sols_fwd
    lines!(ax3d, Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:black)
end
scatter!(ax3d, x_ref[1,1], x_ref[2,1], x_ref[3,1], color=:black, marker=:circle, markersize=12)
display(fig)