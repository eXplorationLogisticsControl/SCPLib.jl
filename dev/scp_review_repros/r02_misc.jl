using Clarabel, JuMP, LinearAlgebra, OrdinaryDiffEq
include("/home/yuri/Documents/codes/jl/SCPLib.jl/src/SCPLib.jl")
section(s) = println("\n==================== ", s, " ====================")
macro tryshow(ex)
    quote
        try
            $(esc(ex))
        catch err
            println("ERROR: ", sprint(showerror, err)[1:min(end,300)])
        end
    end
end

# ---------- (a)(b)(c) forward-backward ----------
section("(a) FB Jacobian sign w.r.t. x_1")
function pend!(dx, x, pu, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1]) + pu.u[1]
    return nothing
end
N = 5; times = collect(LinRange(0.0, 1.0, N))
xr = [0.3 0.8; 0.1 -0.2]; ur = 0.1*ones(1, N-1)
probfb = SCPLib.ContinuousProblem(Clarabel.Optimizer, pend!, nothing, (x,u)->sum(u.^2), times, xr, ur;
    shooting_method=:forwardbackward)
set_silent(probfb.model)
g0 = SCPLib.set_dynamics_cache_forwardbackward!(probfb, xr, ur)
J = probfb.lincache.∇g_dyn
h = 1e-6
fd = zeros(2, 4)
for j in 1:4
    xp = copy(xr); xm = copy(xr); xp[j] += h; xm[j] -= h
    _, gp = SCPLib.get_trajectory_forwardbackward(probfb, xp, ur)
    _, gm = SCPLib.get_trajectory_forwardbackward(probfb, xm, ur)
    fd[:, j] = (gp - gm) / (2h)
end
println("analytic dg/dx1 = ", J[:,1:2]); println("FD       dg/dx1 = ", fd[:,1:2])
println("analytic dg/dxN = ", J[:,3:4]); println("FD       dg/dxN = ", fd[:,3:4])
fdu = zeros(2, N-1)
for j in 1:N-1
    up = copy(ur); um = copy(ur); up[j] += h; um[j] -= h
    _, gp = SCPLib.get_trajectory_forwardbackward(probfb, xr, up)
    _, gm = SCPLib.get_trajectory_forwardbackward(probfb, xr, um)
    fdu[:, j] = (gp - gm) / (2h)
end
println("max |analytic-FD| on u block = ", maximum(abs.(J[:,5:end] - fdu)))
# model-prediction check for a step in x_1 only
dx1 = [0.01, -0.01]
xs = copy(xr); xs[:,1] += dx1
_, gtrue = SCPLib.get_trajectory_forwardbackward(probfb, xs, ur)
println("true g(x1+d)      = ", vec(gtrue))
println("model g0 + J*d    = ", vec(g0) + J[:,1:2]*dx1)
println("correct g0 - Φ d  = ", vec(g0) + fd[:,1:2]*dx1)

section("(b) FB: ξ_dyn decoupled from linearized dynamics")
algo = SCPLib.SCvxStar(2, 2; w0 = 10.0, Δ0 = 1.0, shooting_method=:forwardbackward)
algo.λ_dyn .= [3.0; -5.0]
SCPLib.set_linearized_constraints!(probfb, xr, ur)
SCPLib.set_trust_region_constraints!(algo, probfb, xr, ur)
@constraint(probfb.model, probfb.model[:x][:,1] .== xr[:,1])
SCPLib.solve_convex_subproblem!(algo, probfb, nothing)
println("termination = ", termination_status(probfb.model))
println("value(ξ_dyn) = ", vec(value.(probfb.model[:ξ_dyn])), "  ;  -λ/w = ", -algo.λ_dyn/algo.w)
println("penalty(ξ_dyn) in L = ", SCPLib.penalty(algo, probfb, value.(probfb.model[:ξ_dyn]), nothing, nothing),
        "  (true linearized defect is forced to 0 => should be 0)")

section("(c) FB with w0 = nothing")
probfb2 = SCPLib.ContinuousProblem(Clarabel.Optimizer, pend!, nothing, (x,u)->sum(u.^2), times, copy(xr), copy(ur);
    shooting_method=:forwardbackward)
set_silent(probfb2.model)
algo2 = SCPLib.SCvxStar(2, 2; shooting_method=:forwardbackward)
@tryshow SCPLib.solve!(algo2, probfb2, copy(xr), copy(ur); verbosity=0, maxiter=2)

# ---------- toy problem helpers ----------
lin!(dx, x, pu, t) = (dx[1] = pu.u[1]; nothing)
function toy(; obj=(x,u)->-x[1,2])
    x_ref = zeros(1,2); u_ref = zeros(1,1)
    prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, lin!, nothing, obj, [0.0,1.0], x_ref, u_ref)
    set_silent(prob.model)
    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,2] <= 1.0)
    return prob, x_ref, u_ref
end
countcons(m) = [(F,S,num_constraints(m,F,S)) for (F,S) in list_of_constraint_types(m)]

section("(d) SCvx: model growth (slack vars/cones never deleted)")
prob, xr1, ur1 = toy()
println("before: nvars=", num_variables(prob.model))
algo = SCPLib.SCvx(1, 2; w=1e3, Δ0=0.01)
sol = SCPLib.solve!(algo, prob, xr1, ur1; verbosity=0, maxiter=30)
println("after $(sol.n_iter) iters: nvars=", num_variables(prob.model))
for c in countcons(prob.model); println("   ", c); end

section("(e) ProxLinear: model growth and history misalignment")
prob, xr1, ur1 = toy()
algo = SCPLib.ProxLinear(1e2, 1e2)
iters = Ref(0)
sol = SCPLib.solve!(algo, prob, xr1, ur1; verbosity=0, maxiter=50, callback=(a,s,i,J,c)->(iters[]=i))
println("status=", sol.status, " n_iter=", sol.n_iter, " length(J0)=", length(sol.info[:J0]),
        " last callback iter=", iters[], " nvars=", num_variables(prob.model))
println("info[:J0][end]=", sol.info[:J0][end], "  objective(sol.x)=", -sol.x[1,2])
for c in countcons(prob.model); println("   ", c); end

section("(f) ΔL<0 & ΔJ<0 => ρ≫1, step accepted and TR expanded")
cub!(dx, x, pu, t) = (dx[1] = pu.u[1] + pu.u[1]^3; nothing)
x_ref = zeros(1,2); u_ref = zeros(1,1)
prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, cub!, nothing, (x,u)->u[1,1]^2, [0.0,1.0], x_ref, u_ref)
set_silent(prob.model)
@constraint(prob.model, prob.model[:x][1,1] == 0.0)
@constraint(prob.model, prob.model[:x][1,2] == 1.0)
algo = SCPLib.SCvxStar(1, 2; w0=1e4, Δ0=2.0)
sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity=1, maxiter=1)
println("accept=", sol.info[:accept], " ΔJ=", sol.info[:ΔJ], " Δ after=", algo.tr.Δ)

section("(g) warmstart_dual after re-linearization")
prob, xr1, ur1 = toy()
algo = SCPLib.SCvxStar(1, 2; w0=10.0)
@tryshow begin
    s = SCPLib.solve!(algo, prob, xr1, ur1; verbosity=0, maxiter=5, warmstart_dual=true)
    println("status = ", s.status)
end

section("(h) solve! called twice on same problem")
prob, xr1, ur1 = toy()
algo = SCPLib.SCvxStar(1, 2; w0=10.0)
SCPLib.solve!(algo, prob, xr1, ur1; verbosity=0, maxiter=3)
@tryshow SCPLib.solve!(SCPLib.SCvxStar(1, 2; w0=10.0), prob, zeros(1,2), zeros(1,1); verbosity=0, maxiter=3)

section("(i) μ never decreases; complementarity at 'Optimal' point")
x_ref = fill(-3.0, 1, 2); x_ref[1,1] = 0.0; u_ref = fill(-3.0, 1, 1)
prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, lin!, nothing, (x,u)->-x[1,2], [0.0,1.0], x_ref, u_ref;
    nh=1, h_noncvx=(c,x,u)->[x[1,2]^2 - 4.0])
set_silent(prob.model)
@constraint(prob.model, prob.model[:x][1,1] == 0.0)
@constraint(prob.model, prob.model[:x][1,2] <= 1.0)
@constraint(prob.model, prob.model[:x][1,2] >= -10.0)
algo = SCPLib.SCvxStar(1, 2; nh=1, w0=1.0)
μhist = Float64[]
sol = SCPLib.solve!(algo, prob, x_ref, u_ref; verbosity=0, maxiter=200, callback=(a,s,i,J,c)->push!(μhist, a.μ[1]))
println("status=", sol.status, " x2=", sol.x[1,2], " h(x2)=", sol.x[1,2]^2-4, " final μ=", algo.μ, " w=", algo.w)
println("μ history (unique) = ", unique(round.(μhist, sigdigits=4)))

section("(j) AD-generated eom_aug! with an implicit (autodiff) ODE solver")
probj = SCPLib.ContinuousProblem(Clarabel.Optimizer, pend!, nothing, (x,u)->sum(u.^2), times, [xr[:,1] zeros(2,N-1)], ur;
    ode_method=Rodas5())
@tryshow begin
    _, g = SCPLib.get_trajectory_augmented(probj, [xr[:,1] zeros(2,N-1)], ur)
    println("Rodas5 augmented propagation OK, g[:,1]=", g[:,1])
end
