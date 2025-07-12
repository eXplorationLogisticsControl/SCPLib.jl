Spec out for SCvx
 
- dyad for gnc sim?

- end to end support with AD, for eg with nonlinear constraints, objective function…
- Input should just be eom, nx, nu
- Work either concatenated xu 
- convex constraints added after fact outside
- Impulsive & continuous as separate problem function

SeqCvx

function eom!(dx, x, augmented_p, t)
end

# initialize model
prob = Problem(
- eom!
- objective::Function
- N::Int
- nx
- nu
- g_nl
- h_nl
)

# append convex constraints 
@constraint(prob.model, prob.mode[:x] <= …)