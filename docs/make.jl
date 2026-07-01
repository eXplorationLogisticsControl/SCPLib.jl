"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter

include(joinpath(dirname(@__FILE__), "../src/SCPLib.jl"))


makedocs(
    clean = true,
    build = joinpath(@__DIR__, "build"),
    modules  = [SCPLib],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "https://github.com/eXplorationLogisticsControl/SCPLib.jl/blob/{commit}{path}",
    ),
    sitename = "SCPLib.jl",
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