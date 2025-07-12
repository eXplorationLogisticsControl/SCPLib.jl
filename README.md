# `SCPLib.jl`: Sequential Convex Programming library in Julia


## Installation

For now, `git clone` the repo & add via `pkg> dev ./path/to/SCPLib.jl`.


## Quick start

The main idea is as follows: 

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
    # ...
end

eom_aug! = function (dx_aug, x_aug, params, t)
    # ...
end

x_ref = ...
u_ref = ...
y_ref = nothing   # unless there are variables other than states & controls, set to nothing

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