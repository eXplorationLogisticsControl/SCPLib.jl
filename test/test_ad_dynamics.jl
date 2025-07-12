"""Test AD augmented dynamics"""

using GLMakie
using LinearAlgebra
using OrdinaryDiffEq
using SparseDiffTools
using ForwardDiff
using Test

include(joinpath(@__DIR__, "../src/SCPLib.jl"))


mutable struct ControlParams
    μ::Float64
    u::Vector
    function ControlParams(μ::Float64)
        new(μ, zeros(4))
    end
end


# function test_ad_dynamics()
    μ = 1.215058560962404e-02
    params = ControlParams(μ)

    nx = 6
    nu = 4

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
        return nothing
    end


    function eom(rv, p, t)
        x, y, z = rv[1:3]
        vx, vy, vz = rv[4:6]
        r1 = sqrt( (x+p.μ)^2 + y^2 + z^2 );
        r2 = sqrt( (x-1+p.μ)^2 + y^2 + z^2 );
        drv = [
            vx; vy; vz;
             2*vy + x - ((1-p.μ)/r1^3)*(p.μ+x) + (p.μ/r2^3)*(1-p.μ-x) + p.u[1];
            -2*vx + y - ((1-p.μ)/r1^3)*y - (p.μ/r2^3)*y + p.u[2];
            -((1-p.μ)/r1^3)*z - (p.μ/r2^3)*z + p.u[3];
        ]
        return drv
    end

    # cache for Jacobian
    sd = SymbolicsSparsityDetection()
    adtype = AutoSparse(AutoFiniteDiff())
    cache_jac_A = sparse_jacobian_cache(adtype, sd, (y,x) -> eom!(y, x, params, 1.0), zeros(6), zeros(6))
    # A = zeros(6,6)


    function cr3bp_dfdx(x, p)
        r1vec = [x[1] + p.μ, x[2], x[3]]
        r2vec = [x[1] - 1 + p.μ, x[2], x[3]]
        G1 = (1 - p.μ) / norm(r1vec)^5*(3*r1vec*r1vec' - norm(r1vec)^2*I(3))
        G2 = p.μ / norm(r2vec)^5*(3*r2vec*r2vec' - norm(r2vec)^2*I(3))
        Omega = [0 2 0; -2 0 0; 0 0 0]
        dfdx = [zeros(3,3)                  I(3);
                G1 + G2 + diagm([1,1,0])    Omega]
        return dfdx
    end

    function eom_aug!(dx_aug, x_aug, p, t)
        # compute state derivative with control
        # eom!(dx_aug[1:6], x_aug[1:6], p, t)
        dx_aug[1:6] = eom(x_aug[1:6], p, t)
        
        # ForwardDiff.jacobian!(A, (y,x) -> eom!(y, x, p, t), dx_aug[1:6], x_aug[1:6])
        A = ForwardDiff.jacobian(x -> eom(x, p, t), x_aug[1:6])
        # A = cr3bp_dfdx(x_aug[1:6], p)
        #, cfg::JacobianConfig = JacobianConfig(eom!, dx_aug[1:6], x_aug[1:6]), check=Val{true}())
        # A = ForwardDiff.jacobian(x -> eom!(zeros(6), x, params, t), x_aug[1:6])

        # A = sparse_jacobian(adtype, cache_jac_A, (y,x) -> eom!(y, x, p, t), dx_aug[1:6], x_aug[1:6])
        # A = sparse_jacobian(adtype, cache_jac_A, (y,x) -> eom!(y, x, p, t), similar(x_aug[1:6]), x_aug[1:6])
        B = [zeros(3,4); I(3) zeros(3,1)]
        
        # extract STMs (note: julia is column-major)
        #Phi_A = reshape(x_aug[nx+1:nx*(nx+1)], (nx,nx))'
        Phi_B = reshape(x_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))'  # note: julia is column-major

        # derivatives of Phi_A, Phi_B
        dx_aug[7:42] = reshape((A * reshape(x_aug[7:42],6,6)')', 36)
        dx_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu] = reshape((A * Phi_B + B)', nx*nu)
    end

    # some initial state
    rv0 = [1.0809931218390707E+00,
        0.0000000000000000E+00,
        -2.0235953267405354E-01,
        1.0157158264396639E-14,
        -1.9895001215078018E-01,
        7.2218178975912707E-15]
    tf = 2.3538670417546639E+00

    # set controls
    params.u[1:3] = [-0.02, 0.01, 0.06]
    params.u[4] = norm(params.u[1:3])

    # rv0 += 1e-4 * ones(6)

    # # test Jacobian
    # drv0 = zeros(6)
    # ForwardDiff.jacobian!(A, (y,x) -> eom!(y, x, params, 0.0), drv0, rv0)
    # # A = sparse_jacobian(adtype, cache_jac_A, (y,x) -> eom!(y, x, params, 0.0), drv0, rv0)

    # A_fd = zeros(6,6)
    # h = 1e-7
    # for i in 1:6
    #     drv0_copy = zeros(6)
    #     rv0_copy = copy(rv0)
    #     rv0_copy[i] += h
    #     eom!(drv0_copy, rv0_copy, params, 0.0)
    #     A_fd[:,i] = (drv0_copy - drv0)/h
    # end

    # propagate reference solution
    sol = solve(ODEProblem(eom!,rv0, [0.0, tf], params), Vern8(), reltol=1e-14, abstol=1e-14)

    # use finite-difference to get Phi_A_fd, Phi_B_fd
    Phi_A_fd = zeros(6,6)
    h = 1e-6
    for i = 1:6
        x0_plus = copy(rv0)
        x0_plus[i] += h
        sol_ptrb = solve(ODEProblem(eom!, x0_plus, (0.0, tf), params), Vern8(), reltol=1e-14, abstol=1e-14)

        x0_min = copy(rv0)
        x0_min[i] -= h
        sol_ptrb_min = solve(ODEProblem(eom!, x0_min, (0.0, tf), params), Vern8(), reltol=1e-14, abstol=1e-14)

        Phi_A_fd[:,i] = (sol_ptrb.u[end][1:6] - sol_ptrb_min.u[end][1:6]) / (2*h)
    end

    # propagate numerical sensitivities
    x0_aug = SCPLib.initialize_augmented_state(copy(rv0), 6, 4)
    sol_aug = solve(
        ODEProblem(eom_aug!, x0_aug, [0.0, tf], params), Vern8(), reltol=1e-14, abstol=1e-14
    )
    xf_aug = sol_aug.u[end]
    Phi_A = reshape(xf_aug[nx+1:nx*(nx+1)], (nx,nx))'
    Phi_B = reshape(xf_aug[nx*(nx+1)+1:nx*(nx+1)+nx*nu], (nu,nx))'  # note: julia is column-major

    @show sol.u[end]
    @show sol_aug.u[end][1:6]
    @show sol.u[end] - sol_aug.u[end][1:6]

    # @testset "AD augmented dynamics" begin
    #     @test size(Phi_A) == (nx,nx)
    #     @test size(Phi_B) == (nx,nu)
    # end

    Phi_B_fd = zeros(nx,nu)
    for i in 1:nu
        params_copy = deepcopy(params)
        params_copy.u[i] += h
        sol_forward = solve(
            ODEProblem(eom!, rv0, [0.0, tf], params_copy),
            Tsit5(); reltol = 1e-12, abstol = 1e-12
        )

        params_copy = deepcopy(params)
        params_copy.u[i] -= h
        sol_backward = solve(
            ODEProblem(eom!, rv0, [0.0, tf], params_copy),
            Tsit5(); reltol = 1e-12, abstol = 1e-12
        )

        Phi_B_fd[:,i] = (sol_forward.u[end][1:nx] - sol_backward.u[end][1:nx])/(2*h)
    end

    # @test maximum(abs.(Phi_A - Phi_A_fd)) < 1e-4
    # @test maximum(abs.(Phi_B - Phi_B_fd)) < 1e-4

    # plot trajectory
    fig = Figure()
    ax3d = Axis3(fig[1,1]; aspect=:data)
    lines!(Array(sol)[1,:], Array(sol)[2,:], Array(sol)[3,:], color=:blue)
    lines!(Array(sol_aug)[1,:], Array(sol_aug)[2,:], Array(sol_aug)[3,:], color=:black)
    display(fig)
# end


# test_ad_dynamics()