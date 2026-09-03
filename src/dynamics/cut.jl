"""
Conjugate unscented transform (CUT) for Gaussian densities.

Sigma points follow Adurthi, Singla, and Singh, *Conjugate Unscented
Transformation: Applications to Estimation and Control*, J. Dyn. Sys. Meas.
Control, 2018. Points are generated for N(0, I) and then affinely mapped to
N(μ, Σ).
"""


"""Combinations of `m` indices from `1:n`."""
function _combinations(n::Int, m::Int)
    combs = Vector{Vector{Int}}()
    combo = Vector{Int}(undef, m)
    function rec(start, idx)
        if idx > m
            push!(combs, copy(combo))
            return
        end
        for i in start:(n - m + idx)
            combo[idx] = i
            rec(i + 1, idx + 1)
        end
    end
    rec(1, 1)
    return combs
end


"""Principal axes: columns `±e_i` (`n x 2n`)."""
function _principal_axes(n::Int)
    E = Matrix{Float64}(I, n, n)
    return hcat(E, -E)
end


"""
`m`-th conjugate axes in `n`-D: all sign patterns of `±1` on every combination
of `m` coordinates (`n × 2^m C(n, m)`).
"""
function _conjugate_axes(n::Int, m::Int)
    # 1 << m computes 2^m using bit shift for efficiency: equivalent to 2^m, gives number of sign patterns
    npat = 1 << m
    combs = _combinations(n, m)
    X = zeros(Float64, n, npat * length(combs))
    col = 1
    for comb in combs
        for s in 0:(npat - 1)
            for k in 1:m
                X[comb[k], col] = ((s >> (k - 1)) & 1) == 0 ? -1.0 : 1.0
            end
            col += 1
        end
    end
    return X
end


"""
Scaled conjugate axes `s^n(h)`: nth-conjugate points with one coordinate
scaled by `h` (`n × n 2^n`).
"""
function _scaled_conjugate_axes(n::Int, h::Real)
    g = _conjugate_axes(n, n)
    np = size(g, 2)
    X = Matrix{Float64}(undef, n, n * np)
    for k in 1:n
        cols = ((k - 1) * np + 1):(k * np)
        @views X[:, cols] .= g
        @views X[k, cols] .*= h
    end
    return X
end


function _center_weight(w_rest::AbstractVector)
    w0 = 1 - sum(w_rest)
    return vcat(w0, w_rest)
end


function _cut4_sigma_points(n::Int)
    n >= 2 || throw(ArgumentError("CUT4 requires n ≥ 2, got n = $n"))
    if n == 2
        r1 = 2.6060099476935847
        r2 = 1.190556300661233
        w0 = 0.41553535186548973
        w1 = 0.021681819434216532
        w2 = 0.12443434259941118
        X = hcat(zeros(n), r1 * _principal_axes(n), r2 * _conjugate_axes(n, n))
        w = vcat(w0, fill(w1, 2n), fill(w2, 1 << n))
        return X, w
    end
    r1 = sqrt((n + 2) / 2)
    r2 = sqrt((n + 2) / (n - 2))
    w1 = 4 / (n + 2)^2
    w2 = (n - 2)^2 / (2^n * (n + 2)^2)
    X = hcat(r1 * _principal_axes(n), r2 * _conjugate_axes(n, n))
    w = vcat(fill(w1, 2n), fill(w2, 1 << n))
    return X, w
end


# Radii/weights from Adurthi et al. (2018) Table 27 and nadurthi/CUTpoints.
function _cut6_params(n::Int)
    n == 2 && return (sqrt(6), 1.1147379454, 3.2004125801, 0.0277777777, 0.1302876649, 0.0004653012)
    n == 3 && return (2.3587090379, 1.1198362859, 3.1421303838, 0.0290351301, 0.0633844605, 0.0005195469)
    n == 4 && return (2.2520650012, 1.1260325006, 3.0763780026, 0.0306601632, 0.0306601632, 0.0005898367)
    n == 5 && return (sqrt(9 / 2), 1.1338934190276817, 3.0, 0.03292181069958846, 0.014703360768175577, 0.000685871056241427)
    n == 6 && return (1.9488352799, 1.1445968942, 2.9068006056, 0.0365072564, 0.0069487173, 0.0008288549)
    n == 7 && return (2.5512003554818197, 0.964263097900639, 2.3255766977088315, 0.01269406283896717, 0.004859445930542121, 0.000395089978993786)
    n == 8 && return (sqrt(6), 1.0, sqrt(6), 0.013888888888888888, 0.00234375, 0.0002314814814814815)
    n == 9 && return (2.3439073215294153, 1.023262223053077, 2.5342864499001747, 0.015076391098114098, 0.0011342717964254396, 0.0001572731368706344)
    throw(ArgumentError("CUT6 requires 2 ≤ n ≤ 9, got n = $n"))
end


function _cut6_sigma_points(n::Int)
    r1, r2, r3, w1, w2, w3 = _cut6_params(n)
    # Weights from the sixth-order moment constraint equations (n ≤ 6).
    if n <= 6
        w1 = (8 - n) / r1^6
        w2 = 1 / ((1 << n) * r2^6)
        w3 = 1 / (2 * r3^6)
    end
    m_extra = n <= 6 ? 2 : 3
    Xp = r1 * _principal_axes(n)
    Xn = r2 * _conjugate_axes(n, n)
    Xe = r3 * _conjugate_axes(n, m_extra)
    X = hcat(zeros(n), Xp, Xn, Xe)
    w = _center_weight(vcat(fill(w1, size(Xp, 2)), fill(w2, size(Xn, 2)), fill(w3, size(Xe, 2))))
    return X, w
end


function _cut8_params(n::Int)
    if n == 2
        return (
            r1 = 1 / sqrt(0.23379853497231115),
            r2 = 1 / sqrt(1.3867121461809555),
            r3 = NaN,
            r4 = 1 / sqrt(0.288547926731565),
            r5 = NaN,
            r6 = 1 / sqrt(0.771286446121831),
            h = 3.0,
            w1 = 0.04382264267013926,
            w2 = 0.1405096621714662,
            w3 = 0.0,
            w4 = 0.01240953967762697,
            w5 = 0.0,
            w6 = 0.0009215768861610588,
        )
    elseif n == 3
        return (
            r1 = 1 / sqrt(0.1966319276379789),
            r2 = 1 / sqrt(1.9427321792767849),
            r3 = 1 / sqrt(0.2944016021306629),
            r4 = 1 / sqrt(0.41171525390196356),
            r5 = NaN,
            r6 = 1 / sqrt(0.5866854673078312),
            h = 2.74,
            w1 = 0.024631993437193266,
            w2 = 0.08151009408908164,
            w3 = 0.009767235524166815,
            w4 = 0.00577248937435553,
            w5 = 0.0,
            w6 = 0.000279472936899139,
        )
    elseif n == 4
        return (
            r1 = 1 / sqrt(0.20629093125597198),
            r2 = 1 / sqrt(1.585407549822809),
            r3 = 1 / sqrt(0.2851818320349825),
            r4 = 1 / sqrt(0.5660749628608427),
            r5 = 2.0,
            r6 = 1 / sqrt(0.7889090077901053),
            h = 3.0,
            w1 = 0.01811008737283111,
            w2 = 0.032063273384586845,
            w3 = 0.006614353755080834,
            w4 = 0.003489906522946932,
            w5 = 0.0006510416666666666,
            w6 = 0.00025218336987488566,
        )
    elseif n == 5
        return (
            r1 = 1 / sqrt(0.1866956121576737),
            r2 = 1 / sqrt(1.420294749459945),
            r3 = 1 / sqrt(0.29836021128926843),
            r4 = 1 / sqrt(0.5123685659872401),
            r5 = 2.0,
            r6 = 1 / sqrt(0.8065591548429262),
            h = 3.0,
            w1 = 0.010529034221546607,
            w2 = 0.015144019639537572,
            w3 = 0.0052828996967816825,
            w4 = 0.0010671298950159158,
            w5 = 0.0006510416666666666,
            w6 = 0.00013776017592074394,
        )
    elseif n == 6
        return (
            r1 = 1 / sqrt(0.16666666666666666),
            r2 = 1 / sqrt(1.251685733072443),
            r3 = 1 / sqrt(0.3333333333333333),
            r4 = 1 / sqrt(0.42609204470533507),
            r5 = 2.0,
            r6 = 1 / sqrt(0.8333333333333334),
            h = 3.0,
            w1 = 0.006172839506172839,
            w2 = 0.006913443044833937,
            w3 = 0.004115226337448559,
            w4 = 0.0002183265828666806,
            w5 = 0.0006510416666666666,
            w6 = 0.00007849171328446504,
        )
    end
    throw(ArgumentError("CUT8 requires 2 ≤ n ≤ 6, got n = $n"))
end

function _cut8_sigma_points(n::Int)
    p = _cut8_params(n)
    Xc = zeros(n, 1)
    X1 = p.r1 * _principal_axes(n)
    X2 = p.r2 * _conjugate_axes(n, n)
    X4 = p.r4 * _conjugate_axes(n, n)
    X6 = p.r6 * _scaled_conjugate_axes(n, p.h)

    if n == 2
        X = hcat(Xc, X1, X2, X4, X6)
        w = _center_weight(vcat(
            fill(p.w1, size(X1, 2)),
            fill(p.w2, size(X2, 2)),
            fill(p.w4, size(X4, 2)),
            fill(p.w6, size(X6, 2)),
        ))
        return X, w
    end

    X3 = p.r3 * _conjugate_axes(n, 2)
    if n == 3
        X = hcat(Xc, X1, X2, X4, X3, X6)
        w = _center_weight(vcat(
            fill(p.w1, size(X1, 2)),
            fill(p.w2, size(X2, 2)),
            fill(p.w4, size(X4, 2)),
            fill(p.w3, size(X3, 2)),
            fill(p.w6, size(X6, 2)),
        ))
        return X, w
    end

    X5 = p.r5 * _conjugate_axes(n, 3)
    X = hcat(Xc, X1, X2, X4, X3, X5, X6)
    w = _center_weight(vcat(
        fill(p.w1, size(X1, 2)),
        fill(p.w2, size(X2, 2)),
        fill(p.w4, size(X4, 2)),
        fill(p.w3, size(X3, 2)),
        fill(p.w5, size(X5, 2)),
        fill(p.w6, size(X6, 2)),
    ))
    return X, w
end


"""
    cut_sigma_points(n, degree=4) -> (X0, w)

Generate standard-normal CUT sigma points. `X0` is `n × N` (columns are points) and `w`
is a length-`N` weight vector.

`degree` must be 4, 6, or 8. Supported dimensions: CUT4 `n ≥ 2`, CUT6
`2 ≤ n ≤ 9`, CUT8 `2 ≤ n ≤ 6`.
"""
function cut_sigma_points(n::Int, degree::Int=4)
    n >= 2 || throw(ArgumentError("CUT requires dimension n ≥ 2, got n = $n"))
    if degree == 4
        return _cut4_sigma_points(n)
    elseif degree == 6
        return _cut6_sigma_points(n)
    elseif degree == 8
        return _cut8_sigma_points(n)
    else
        throw(ArgumentError("Degree must be 4, 6, or 8, got $degree"))
    end
end


"""
    cut(mu, Sigma, f_nonlinear; degree=4) -> (mu_out, Sigma_out)

Propagate a Gaussian `N(mu, Sigma)` through `f_nonlinear` with the conjugate
unscented transform of the given `degree` (4, 6, or 8).

`f_nonlinear(x::AbstractVector)` may return a vector whose length differs from
`length(mu)`. The output dimension is inferred from the first mapped point.
"""
function cut(
    mu::AbstractVector,
    Sigma::AbstractMatrix,
    f_nonlinear::Function;
    degree::Int=4,
)
    n = length(mu)
    size(Sigma) == (n, n) || throw(ArgumentError(
        "Sigma must be $(n)x$(n), got $(size(Sigma))"))

    X0, w = cut_sigma_points(n, degree)
    X = sqrt(Hermitian(Sigma)) * X0 .+ mu

    N = length(w)
    z1 = collect(f_nonlinear(@view X[:, 1]))
    nz = length(z1)
    Z = Matrix{float(eltype(z1))}(undef, nz, N)
    Z[:, 1] = z1
    for i in 2:N
        Z[:, i] = f_nonlinear(@view X[:, i])
    end

    mu_out = Z * w
    Δ = Z .- mu_out
    Sigma_out = Δ * Diagonal(w) * Δ'
    return mu_out, Sigma_out
end
