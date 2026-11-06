# Repro: max(h_ref,0) linearization + monotone mu => false convergence (SCvx*, ProxLinear)
using Clarabel, JuMP, LinearAlgebra, OrdinaryDiffEq
include("/home/yuri/Documents/codes/jl/SCPLib.jl/src/SCPLib.jl")

eom!(dx, x, pu, t) = (dx[1] = pu.u[1]; nothing)

function make(; nh_on=true, convex_instead=false)
    x_ref = zeros(1,2); u_ref = zeros(1,1)
    obj(x,u) = -x[1,2]                      # maximise x2
    h(cache,x,u) = [x[1,2] - 2.0]           # x2 <= 2 : NEVER active (x2 <= 1 enforced convexly)
    kw = nh_on ? (nh=1, h_noncvx=h) : NamedTuple()
    prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, eom!, nothing, obj, [0.0,1.0], x_ref, u_ref; kw...)
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,2] <= 1.0)
    if convex_instead
        @constraint(prob.model, prob.model[:x][1,2] <= 2.0)
    end
    return prob, x_ref, u_ref
end

println("=== SCvx*, h as nonconvex inequality, w0 = nothing (auto) ===")
prob, xr, ur = make()
algo = SCPLib.SCvxStar(1, 2; nh=1)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) iters=$(sol.n_iter) x2=$(sol.x[1,2]) (true optimum 1.0) w=$(algo.w) mu=$(algo.μ)")

println("=== SCvx*, h as nonconvex inequality, w0 = 10 ===")
prob, xr, ur = make()
algo = SCPLib.SCvxStar(1, 2; nh=1, w0=10.0)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) iters=$(sol.n_iter) x2=$(sol.x[1,2]) (true optimum 1.0) w=$(algo.w)")

println("=== SCvx*, same constraint given as convex (control) ===")
prob, xr, ur = make(nh_on=false, convex_instead=true)
algo = SCPLib.SCvxStar(1, 2; w0=10.0)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) iters=$(sol.n_iter) x2=$(sol.x[1,2])")

println("=== ProxLinear, h as nonconvex inequality ===")
prob, xr, ur = make()
algo = SCPLib.ProxLinear(1e2, 1.0)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) x2=$(sol.x[1,2]) (true optimum 1.0)")

println("=== ProxLinear, convex control ===")
prob, xr, ur = make(nh_on=false, convex_instead=true)
algo = SCPLib.ProxLinear(1e2, 1.0)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) x2=$(sol.x[1,2])")

println("=== SCvx (L1), h as nonconvex inequality ===")
prob, xr, ur = make()
algo = SCPLib.SCvx(1, 2; w=1e3)
sol = SCPLib.solve!(algo, prob, xr, ur; verbosity=0, maxiter=200)
println("status=$(sol.status) iters=$(sol.n_iter) x2=$(sol.x[1,2]) (true optimum 1.0)")
