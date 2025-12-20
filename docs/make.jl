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
    # options
    pages = [
		"Home" => "index.md",
    ],
)