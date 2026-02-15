"""
Make documentation with Documenter.jl
"""

using Documenter

include(joinpath(dirname(@__FILE__), "../src/SCPLib.jl"))


makedocs(
    clean = false,
    build = joinpath(dirname(@__FILE__), "build"),
    modules  = [SCPLib],
    format   = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "SCPLib.jl",
    checkdocs=:exports,
    # options
    pages = [
      "Home" => "index.md",
      "API" => "api.md",
    ],
)