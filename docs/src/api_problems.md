# API: Problems & Dynamics

## Continuous problems

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "continuous_problem.jl",
]
```

## Impulsive problems

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "impulsive_problem.jl",
]
```

## Mixed problems

Continuous controls on nodes `1:N-1` and impulsive controls on nodes `1:N`.
See `MixedProblem`.

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "mixed_problem.jl",
]
```

## Linear dynamics

For linear (affine) dynamics, `set_linear_dynamics!` computes the multiple-shooting matrices once,
so that `solve!` does no ODE propagation. Non-convex path and boundary constraints are given as
usual through `g_noncvx`/`h_noncvx`.

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "linear_dynamics.jl",
]
```

## Equations of motion

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "dynamics.jl",
]
```

## Linearization cache

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "linearization.jl",
]
```

## Nonlinear constraints

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "nlconstraints.jl",
]
```

## Utilities

```@autodocs
Modules = [SCPLib]
Order   = [:function, :type, :struct]
Pages   = [
  "utils.jl",
]
```
