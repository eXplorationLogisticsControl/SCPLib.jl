using Clarabel, JuMP, LinearAlgebra, OrdinaryDiffEq
include("/home/yuri/Documents/codes/jl/SCPLib.jl/src/SCPLib.jl")
function pend!(dx, x, pu, t)
    dx[1] = x[2]; dx[2] = -sin(x[1]) + pu.u[1]; return nothing
end
N = 5; times = collect(LinRange(0.0, 1.0, N)); x0 = [0.3 0.0 0.0 0.0 0.0; 0.1 0 0 0 0]; ur = 0.1*ones(1,N-1)
prob = SCPLib.ContinuousProblem(Clarabel.Optimizer, pend!, nothing, (x,u)->sum(u.^2), times, x0, ur; ode_method=Rodas5())
_, g = SCPLib.get_trajectory(prob, x0, ur); println("plain eom! with Rodas5 OK: ", g[:,1])
try
    SCPLib.get_trajectory_augmented(prob, x0, ur)
catch err
    s = sprint(showerror, err); i = findfirst("Dual", s)
    println("augmented AD eom with Rodas5 FAILS; excerpt: ", isnothing(i) ? s[1:200] : s[max(1,i[1]-300):min(end,i[1]+300)])
end
prob2 = SCPLib.ContinuousProblem(Clarabel.Optimizer, pend!, nothing, (x,u)->sum(u.^2), times, x0, ur; ode_method=Rodas5(autodiff=false))
_, g2 = SCPLib.get_trajectory_augmented(prob2, x0, ur); println("augmented with Rodas5(autodiff=false) OK")
