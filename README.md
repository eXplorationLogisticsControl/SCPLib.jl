# `SCPLib.jl`: Sequential Convex Programming library in Julia

The optimal control problem (OCP) of interest is of the form

```math
```

## Installation

For now, `git clone` the repo & add via `pkg> dev ./path/to/SCPLib.jl`.


## Quick start

We first need to define a few things:

- a parameter `mutable struct`, which includes a vector `u` (the control vector),
- the controlled equations of motion `eom!`, which takes as parameter the aforementioned `mutable struct`,
- (optionally) the augmented equations of motion which propagates the state together with the STM's $\Phi_A$ and $\Phi_B$,
- an array of time-stamps corresponding to the discretized nodes, and
- initial guesses for state & control histories, `x_ref` and `u_ref`; if the problem also has other variables `y`, then we also need initial guess for those, i.e. `y_ref`.


```julia
# 0. define dynamics, define initial reference (i.e. initial guess), etc.
nx = 6   # state dimension
nu = 4   # control dimension
mutable struct MyODEParams
    p           # ODE Parameters
    u::Vector   # control vector, length `nu`
end

params = MyODEParams(p, zeros(nu))

eom! = function (dx, x, params, t)
    # compute derivative of state 
    # ...
end

eom_aug! = function (dx_aug, x_aug, params, t)
    # compute derivative of state & Phi_A & Phi_B
    # ...
end

N = 60                          # number of nodes
times = LinRange(0.0, tf, N)    # time-stamp at nodes
x_ref = ...                     # initial guess for state history
u_ref = ...                     # initial guess for control history
y_ref = nothing                 # unless there are variables other than states & controls, set to nothing
```

Now we can use `SCPLib` to construct & solve our optimal control problem!

```julia
# 1. instantiate problem    
prob = SCPLib.ContinuousProblem(
    Clarabel.Optimizer,
    eom!,
    params,
    objective,
    times,
    x_ref,
    u_ref,
    y_ref;
    eom_aug! = eom_aug!,
)
set_silent(prob.model)

# 2. instantiate algorithm
algo = SCPLib.SCvxStar(nx, N; w0 = 1e4)   # as an example, setting `w0` to a non-default value

# 3. solve problem
solution = SCPLib.solve!(algo, prob, x_ref, u_ref, y_ref; maxiter = 100)
```