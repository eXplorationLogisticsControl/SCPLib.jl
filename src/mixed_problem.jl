"""Mixed continuous and impulsive SCP problem.

Continuous controls live on nodes `1:N-1`. Impulsive controls live on nodes `1:N`,
including the arrival burn at node `N`. On nodes `1:N-1` the decision vector is
stacked `[continuous; impulsive]`. A square reference of size `(nuc + nui, N)`
stores `NaN` in `u[1:nuc, N]`. Propagation, linearization, the objective call
sites in this file, and trust-region updates do not read those entries.

The coast sees only the continuous slice. `dfdu(x, u_impulsive, t)` returns
`(nx, nui)`. Each segment jumps, then flows. `Φ_A` starts at `I` after the jump.
After the coast, `Φ_B = [Φ_B_continuous  Φ_A * dfdu]`. A user `eom_aug!` integrates
`Φ_A` and the continuous `Φ_B` only (`nu` in that ODE is `nuc`).
"""


mutable struct MixedProblem <: OptimalControlProblem
    nx::Int
    nu::Int
    nuc::Int
    nui::Int
    N::Int
    Nu::Int

    ng::Int
    nh::Int

    eom!::Function
    eom_aug!::Function
    dfdu::Function
    params

    times::Union{Vector,LinRange}

    objective::Function
    g_noncvx::Union{Function,Nothing}
    ∇g_noncvx::Union{Function,Nothing}
    h_noncvx::Union{Function,Nothing}
    ∇h_noncvx::Union{Function,Nothing}

    model::Model
    model_nl_references::Vector{Symbol}

    lincache::AbstractLinearizationCache

    ode_ensemble_method
    ode_method
    ode_reltol::Float64
    ode_abstol::Float64

    fun_get_trajectory::Union{Function,Nothing}
    set_dynamics_cache!::Union{Function,Nothing}
    set_linearized_constraints!::Union{Function,Nothing}

    u_bias::Matrix
    shooting_method::Symbol
end


function Base.show(io::IO, prob::MixedProblem)
    @printf("Mixed optimal control problem\n")
    @printf("  Shooting method : %s\n", prob.shooting_method)
    @printf("  nx              : %d\n", prob.nx)
    @printf("  nuc             : %d\n", prob.nuc)
    @printf("  nui             : %d\n", prob.nui)
    @printf("  nu              : %d\n", prob.nu)
    @printf("  N               : %d\n", prob.N)
    @printf("  ng              : %d\n", prob.ng)
    @printf("  nh              : %d\n", prob.nh)
    @printf("  ODE ensemble    : %s\n", prob.ode_ensemble_method)
    @printf("  ODE method      : %s\n", prob.ode_method)
    @printf("  ODE reltol      : %1.4e\n", prob.ode_reltol)
    @printf("  ODE abstol      : %1.4e\n", prob.ode_abstol)
end


"""Length of the flattened control decision vector, excluding `u[1:nuc, N]`."""
mixed_u_flat_length(nuc::Int, nui::Int, N::Int) = nuc * (N - 1) + nui * N


function assert_inactive_control_nan(A, nuc::Int, N::Int, label::AbstractString)
    nuc == 0 && return
    @assert size(A) == (size(A, 1), N) "$(label) must have N columns"
    @assert all(isnan, @view(A[1:nuc, N])) "$(label)[1:$(nuc), $(N)] must be NaN"
    if N > 1
        @assert all(isfinite, @view(A[1:nuc, 1:N-1])) "$(label) continuous nodes 1:N-1 must be finite"
    end
    @assert all(isfinite, @view(A[nuc+1:end, :])) "$(label) impulsive entries must be finite"
    return
end


"""Numeric control values. Mixed problems write `NaN` into the unused continuous column `N`."""
function control_values(prob::OptimalControlProblem)
    return value.(prob.model[:u])
end


function control_values(prob::MixedProblem)
    nuc, nu, N = prob.nuc, prob.nu, prob.N
    U = Matrix{Float64}(undef, nu, N)
    u = prob.model[:u]
    for k in 1:N-1
        for i in 1:nu
            U[i, k] = value(u[i, k])
        end
    end
    if nuc > 0
        U[1:nuc, N] .= NaN
    end
    for i in nuc+1:nu
        U[i, N] = value(u[i, N])
    end
    return U
end


function initial_control_matrix(prob::OptimalControlProblem, Nu::Int)
    return zeros(prob.nu, Nu)
end


function initial_control_matrix(prob::MixedProblem, Nu::Int)
    u = zeros(prob.nu, Nu)
    if prob.nuc > 0 && Nu >= prob.N
        u[1:prob.nuc, prob.N] .= NaN
    end
    return u
end


function mask_unused_control_trust_region!(::Any, ::OptimalControlProblem)
    return
end


function mask_unused_control_trust_region!(algo, prob::MixedProblem)
    if !algo.use_trustregion_control || prob.nuc == 0
        return
    end
    Δ = algo.tr_u.Δ
    if size(Δ, 1) >= prob.nuc && size(Δ, 2) >= prob.N
        Δ[1:prob.nuc, prob.N] .= NaN
    end
    return
end


function append_control_deltas!(Δvars, prob::OptimalControlProblem, u_ref)
    append!(Δvars, reshape(prob.model[:u] - u_ref, prod(size(prob.model[:u]))))
    return Δvars
end


function append_control_deltas!(Δvars, prob::MixedProblem, u_ref)
    u = prob.model[:u]
    nuc, nu, N = prob.nuc, prob.nu, prob.N
    for k in 1:N-1
        for i in 1:nu
            push!(Δvars, u[i, k] - u_ref[i, k])
        end
    end
    for i in nuc+1:nu
        push!(Δvars, u[i, N] - u_ref[i, N])
    end
    return Δvars
end


function flattened_delta(prob::OptimalControlProblem, x_ref, u_ref)
    return stack_flatten_variables(prob, prob.model[:x] - x_ref, prob.model[:u] - u_ref)
end


function flattened_delta(prob::MixedProblem, x_ref, u_ref)
    xΔ = reshape(prob.model[:x] - x_ref, prob.nx * prob.N)
    u = prob.model[:u]
    nuc, nu, N = prob.nuc, prob.nu, prob.N
    n_flat = mixed_u_flat_length(nuc, prob.nui, N)
    Δu = Vector{AffExpr}(undef, n_flat)
    idx = 0
    for k in 1:N-1
        for i in 1:nu
            idx += 1
            Δu[idx] = u[i, k] - u_ref[i, k]
        end
    end
    for i in nuc+1:nu
        idx += 1
        Δu[idx] = u[i, N] - u_ref[i, N]
    end
    return [xΔ; Δu]
end


"""Trust-region rows that exist at node `k`. Column `N` is impulsive only."""
function _mixed_trust_rows(prob::MixedProblem, k::Int)
    if k == prob.N
        return prob.nuc+1:prob.nu
    else
        return 1:prob.nu
    end
end


function add_control_trust_region!(algo::SCPAlgorithm, prob::MixedProblem, u_ref)
    Nu = size(u_ref, 2)
    @constraint(prob.model, constraint_trust_region_u_lb[k in 1:Nu, i in _mixed_trust_rows(prob, k)],
        -(prob.model[:u][i, k] - u_ref[i, k]) <= algo.tr_u.Δ[i, k])
    @constraint(prob.model, constraint_trust_region_u_ub[k in 1:Nu, i in _mixed_trust_rows(prob, k)],
        prob.model[:u][i, k] - u_ref[i, k] <= algo.tr_u.Δ[i, k])
    return
end


function _fill_mixed_slice!(dest, src, bias, rows, k)
    dest .= @view(src[rows, k]) .+ @view(bias[rows, k])
    return dest
end


"""Augmented state after the impulsive jump: `[x + dfdu*u_i; vec(I); 0]`, with `Φ_B` width `nuc`."""
function init_mixed_dynamics_xaug(t0::Real, x0, u_i, nx::Int, nuc::Int, dfdu::Function)
    x_jump = x0 + dfdu(x0, u_i, t0) * u_i
    return [x_jump;
            reshape(I(nx), nx^2);
            zeros(nx * nuc)]
end


function get_trajectory(prob::MixedProblem, x_ref, u_ref)
    g_dynamics = zeros(prob.nx, prob.N - 1)
    nuc, nui = prob.nuc, prob.nui
    uc_pool = make_u_pool(nuc, prob.N - 1)
    ui_pool = make_u_pool(nui, prob.N - 1)
    p_placeholder = dynamics_input(prob.params, uc_pool[1])
    prob_func = function(ode_problem, ctx_or_i, repeat = nothing)
        i = ensemble_sim_id(ctx_or_i)
        uc = _fill_mixed_slice!(uc_pool[i], u_ref, prob.u_bias, 1:nuc, i)
        ui = _fill_mixed_slice!(ui_pool[i], u_ref, prob.u_bias, nuc+1:nuc+nui, i)
        x_jump = x_ref[:, i] + prob.dfdu(x_ref[:, i], ui, prob.times[i]) * ui
        remake(ode_problem,
            u0 = x_jump,
            tspan = (prob.times[i], prob.times[i+1]),
            p = dynamics_input(prob.params, uc),
        )
    end
    base_ode_problem = ODEProblem(
        prob.eom!,
        x_ref[:, 1],
        [0.0, 1.0],
        p_placeholder,
    )
    ensemble_prob = EnsembleProblem(base_ode_problem, prob_func = prob_func)
    sols = solve(
        ensemble_prob,
        prob.ode_method,
        prob.ode_ensemble_method;
        trajectories = prob.N - 1,
        reltol = prob.ode_reltol,
        abstol = prob.ode_abstol,
        verbose = ode_verbose(),
    )
    for (k, sol) in enumerate(ensemble_trajectories(sols))
        g_dynamics[:, k] = x_ref[:, k+1] - sol.u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


function get_trajectory_augmented(prob::MixedProblem, x_ref, u_ref)
    g_dynamics = zeros(prob.nx, prob.N - 1)
    nuc, nui = prob.nuc, prob.nui
    uc_pool = make_u_pool(nuc, prob.N - 1)
    ui_pool = make_u_pool(nui, prob.N - 1)
    p_placeholder = dynamics_input(prob.params, uc_pool[1])
    xaug0 = init_mixed_dynamics_xaug(prob.times[1], x_ref[:, 1], zeros(nui), prob.nx, nuc, prob.dfdu)
    prob_func = function(ode_problem, ctx_or_i, repeat = nothing)
        i = ensemble_sim_id(ctx_or_i)
        uc = _fill_mixed_slice!(uc_pool[i], u_ref, prob.u_bias, 1:nuc, i)
        ui = _fill_mixed_slice!(ui_pool[i], u_ref, prob.u_bias, nuc+1:nuc+nui, i)
        xaug = init_mixed_dynamics_xaug(prob.times[i], x_ref[:, i], ui, prob.nx, nuc, prob.dfdu)
        remake(ode_problem,
            u0 = xaug,
            tspan = (prob.times[i], prob.times[i+1]),
            p = dynamics_input(prob.params, uc),
        )
    end
    base_ode_problem = ODEProblem(
        prob.eom_aug!,
        xaug0,
        [0.0, 1.0],
        p_placeholder,
    )
    ensemble_prob = EnsembleProblem(base_ode_problem, prob_func = prob_func)
    sols = solve(
        ensemble_prob,
        prob.ode_method,
        prob.ode_ensemble_method;
        trajectories = prob.N - 1,
        reltol = prob.ode_reltol,
        abstol = prob.ode_abstol,
        verbose = ode_verbose(),
    )
    for (k, sol) in enumerate(ensemble_trajectories(sols))
        g_dynamics[:, k] = x_ref[:, k+1] - sol.u[end][1:prob.nx]
    end
    return sols, g_dynamics
end


"""
Construct a mixed continuous/impulsive optimal control problem.

# Arguments
- `optimizer`: optimizer for the JuMP model
- `eom!`: coast dynamics `(dx, x, (; params, u), t)`. `u` has length `nuc` (continuous slice only).
- `params`: immutable physical parameters (must not contain segment control)
- `objective`: `(x, u) -> J`. `u` is `(nuc + nui)` by `N` and `u[1:nuc, N]` is `NaN` on numeric arrays.
- `times`: node times. Variable time is encoded in the dynamics, not by editing `times`.
- `x_ref`: `(nx, N)` reference state
- `u_ref`: `(nuc + nui, N)` reference control. `u_ref[1:nuc, N]` must be `NaN`.
- `nuc`: number of continuous controls
- `nui`: number of impulsive controls
- `dfdu`: `(x, u_impulsive, t) -> (nx, nui)` jump map. The jump is `x + dfdu * u_impulsive` at the start of each segment. It is not applied to column `N` inside the propagator.
- `eom_aug!`: optional augmented coast. State layout is `[x; vec(Φ_A); vec(Φ_B_continuous)]` with `Φ_B` width `nuc`. The impulse is applied before this ODE and is not part of `eom_aug!`.
"""
function MixedProblem(
    optimizer,
    eom!::Function,
    params,
    objective::Function,
    times,
    x_ref,
    u_ref;
    nuc::Int,
    nui::Int,
    dfdu::Function,
    eom_aug! = nothing,
    ng::Int = 0,
    g_noncvx::Union{Function,Nothing} = nothing,
    ∇g_noncvx::Union{Function,Nothing} = nothing,
    nh::Int = 0,
    h_noncvx::Union{Function,Nothing} = nothing,
    ∇h_noncvx::Union{Function,Nothing} = nothing,
    ode_ensemble_method = SciMLBase.EnsembleSerial(),
    ode_method = Tsit5(),
    ode_reltol::Float64 = 1e-12,
    ode_abstol::Float64 = 1e-12,
    fun_get_trajectory::Union{Function,Nothing} = nothing,
    set_dynamics_cache!::Union{Function,Nothing} = nothing,
    set_linearized_constraints!::Union{Function,Nothing} = nothing,
    lincache = nothing,
    u_bias::Union{Matrix,Nothing} = nothing,
    shooting_method::Symbol = :multiple,
)
    @assert shooting_method == :multiple "MixedProblem supports multiple shooting only"
    @assert nuc >= 0 && nui >= 0
    @assert nuc + nui > 0 "MixedProblem needs at least one control channel"

    N = length(times)
    @assert N >= 2
    @assert size(x_ref, 2) == N
    nx = size(x_ref, 1)
    nu = nuc + nui
    @assert size(u_ref, 1) == nu "nu = nuc + nui must equal the row count of u_ref"
    @assert size(u_ref, 2) == N "u_ref must be (nuc + nui, N), with NaN in the continuous arrival column"
    assert_inactive_control_nan(u_ref, nuc, N, "u_ref")

    if isnothing(set_dynamics_cache!)
        u_i0 = nui == 0 ? zeros(0) : u_ref[nuc+1:end, 1]
        B = dfdu(x_ref[:, 1], u_i0, times[1])
        @assert size(B) == (nx, nui) "Size of output from dfdu is not (nx, nui)"
    end

    if isnothing(eom_aug!)
        eom_aug! = get_continuous_augmented_eom(eom!, params, nx, nuc)
    end

    n_u_flat = mixed_u_flat_length(nuc, nui, N)
    if isnothing(lincache)
        lincache = MultipleShootingCache(nx, nu, N, N, ng, nh, n_u_flat)
    end

    if !isnothing(g_noncvx) && isnothing(∇g_noncvx)
        ∇g_noncvx = function (x, u)
            return ForwardDiff.jacobian(z -> g_noncvx(lincache, unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u))
        end
    end

    if !isnothing(h_noncvx) && isnothing(∇h_noncvx)
        ∇h_noncvx = function (x, u)
            return ForwardDiff.jacobian(z -> h_noncvx(lincache, unpack_flattened_variables(prob, z)...),
                                        stack_flatten_variables(prob, x, u))
        end
    end

    model_nl_references = [:constraint_dynamics,
                           :constraint_trust_region_x_lb,
                           :constraint_trust_region_x_ub]

    if isnothing(u_bias)
        u_bias = zeros(nu, N)
        if nuc > 0
            u_bias[1:nuc, N] .= NaN
        end
    else
        @assert size(u_bias) == (nu, N)
        assert_inactive_control_nan(u_bias, nuc, N, "u_bias")
    end

    prob = MixedProblem(
        nx,
        nu,
        nuc,
        nui,
        N,
        N,
        ng,
        nh,
        eom!,
        eom_aug!,
        dfdu,
        params,
        times,
        objective,
        g_noncvx,
        ∇g_noncvx,
        h_noncvx,
        ∇h_noncvx,
        Model(optimizer),
        model_nl_references,
        lincache,
        ode_ensemble_method,
        ode_method,
        ode_reltol,
        ode_abstol,
        fun_get_trajectory,
        set_dynamics_cache!,
        set_linearized_constraints!,
        u_bias,
        shooting_method,
    )

    @variable(prob.model, x[i=1:nx, k=1:N])
    if nuc > 0
        @variable(prob.model, uc[i=1:nuc, k=1:N-1])
    end
    if nui > 0
        @variable(prob.model, ui[i=1:nui, k=1:N])
    end
    @variable(prob.model, ξ_dyn[i=1:nx, k=1:N-1])

    u = Matrix{Any}(undef, nu, N)
    for k in 1:N-1
        for i in 1:nuc
            u[i, k] = prob.model[:uc][i, k]
        end
        for i in 1:nui
            u[nuc+i, k] = prob.model[:ui][i, k]
        end
    end
    for i in 1:nuc
        u[i, N] = NaN
    end
    for i in 1:nui
        u[nuc+i, N] = prob.model[:ui][i, N]
    end
    prob.model[:u] = u

    if ng > 0
        @variable(prob.model, ξ[i=1:ng])
        push!(prob.model_nl_references, :constraint_g_noncvx)
    end

    if nh > 0
        @variable(prob.model, ζ[i=1:nh] >= 0.0)
        push!(prob.model_nl_references, :constraint_h_noncvx)
    end
    return prob
end


function stack_flatten_variables(prob::MixedProblem, x, u)
    nx, nuc, nui, N = prob.nx, prob.nuc, prob.nui, prob.N
    xz = reshape(x, nx * N)
    z = similar(xz, nx * N + mixed_u_flat_length(nuc, nui, N))
    z[1:nx*N] = xz
    idx = nx * N
    for k in 1:N-1
        if nuc > 0
            z[idx+1:idx+nuc] = u[1:nuc, k]
            idx += nuc
        end
        if nui > 0
            z[idx+1:idx+nui] = u[nuc+1:end, k]
            idx += nui
        end
    end
    if nui > 0
        z[idx+1:idx+nui] = u[nuc+1:end, N]
    end
    return z
end


function unpack_flattened_variables(prob::MixedProblem, z)
    nx, nuc, nui, N, nu = prob.nx, prob.nuc, prob.nui, prob.N, prob.nu
    T = eltype(z)
    x = reshape(z[1:nx*N], nx, N)
    u = Matrix{T}(undef, nu, N)
    idx = nx * N
    for k in 1:N-1
        if nuc > 0
            u[1:nuc, k] = z[idx+1:idx+nuc]
            idx += nuc
        end
        if nui > 0
            u[nuc+1:end, k] = z[idx+1:idx+nui]
            idx += nui
        end
    end
    if nuc > 0
        u[1:nuc, N] .= T(NaN)
    end
    if nui > 0
        u[nuc+1:end, N] = z[idx+1:idx+nui]
    end
    return x, u
end
