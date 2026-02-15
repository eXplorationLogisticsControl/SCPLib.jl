"""
Make documentation with Documenter.jl
"""

using Documenter

include(joinpath(dirname(@__FILE__), "../src/SCPLib.jl"))


makedocs(
  clean = false,
  build = dirname(@__FILE__),
  modules  = [SCPLib],
  format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  sitename = "SCPLib.jl",
  checkdocs=:exports,
  # options
  pages = [
    "Home" => "index.md",
    "Tutorials" => Any[
      "Basic OCP" => "ocp_basic_cr3bp.md",
      "Non-convex path constraints" => "ocp_ncvx.md",
    ],
    "Examples" => Any[
      "Dionysus problem" => "examples/example_dionysus.md",
    ],
    "API" => Any[
      "API: Algorithms" => "api_algorithms.md",
      "API: Problems & Dynamics" => "api_problems.md",
    ],
  ]
)