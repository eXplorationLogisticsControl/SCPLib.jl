"""
Make documentation with Documenter.jl
"""

import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using Literate

include(joinpath(dirname(@__FILE__), "../src/SCPLib.jl"))


# Convert Literate.jl sources in `docs/literate/` into markdown pages under `docs/src/`,
# keeping the relative path (e.g. `literate/advanced/x.jl` -> `src/advanced/x.md`).
# The code is not executed here: blocks are emitted as plain `julia` code fences, and figures are
# committed under `docs/src/assets/` (regenerate them by running the Literate source as a script).
literate_dir = joinpath(@__DIR__, "literate")
for (root, _, files) in walkdir(literate_dir)
    for file in filter(endswith(".jl"), files)
        outdir = joinpath(@__DIR__, "src", relpath(root, literate_dir))
        Literate.markdown(
            joinpath(root, file), outdir;
            flavor = Literate.DocumenterFlavor(),
            codefence = "```julia" => "```",
            credit = true,
        )
    end
end


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
        "Algorithms" => Any[
          "SCvx*" => "algorithms/scvxstar.md",
          "SCvx" => "algorithms/scvx.md",
        ],
        "Advanced" => Any[
          "User-defined linearized constraints" => "advanced/set_linearized_constraints.md",
          "Impulsive control with variable time" => "advanced/impulsive_control_variable_time.md",
        ],
        "Examples" => Any[
          "Brachistochrone" => "examples/example_brachistochrone.md",
          "Hypersensitive" => "examples/example_hypersensitive.md",
          "Supersonic time-to-climb" => "examples/example_supersonic.md",
          "Space shuttle reentry" => "examples/example_spaceshuttle.md",
          "Dionysus problem" => "examples/example_dionysus.md",
        ],
        "API" => Any[
          "API: Algorithms" => "api_algorithms.md",
          "API: Problems & Dynamics" => "api_problems.md",
        ],
    ]
)