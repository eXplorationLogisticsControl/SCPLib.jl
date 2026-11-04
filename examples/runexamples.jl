"""
    requires_gurobi(path) -> Bool

Return whether `path` loads Gurobi. Commented `using` lines are ignored.
"""
function requires_gurobi(path::AbstractString)
    for line in eachline(path)
        occursin(r"^\s*using\b.*\bGurobi\b", line) && return true
    end
    return false
end

"""
    runexamples(dir::AbstractString=@__DIR__; project::AbstractString=dir, skip_gurobi::Bool=false)

Run every Julia example under `dir`, recursively. Jupyter notebooks are skipped.
Each script runs in its own Julia process, using `project` as the environment,
so type and function definitions do not collide across examples.
When `skip_gurobi` is true, scripts that load Gurobi are skipped.
"""
function runexamples(dir::AbstractString=@__DIR__; project::AbstractString=dir, skip_gurobi::Bool=false)
    scripts = String[]
    skipped = String[]
    for (root, _, files) in walkdir(dir)
        for file in files
            endswith(file, ".jl") || continue
            path = abspath(joinpath(root, file))
            path == abspath(@__FILE__) && continue
            if skip_gurobi && requires_gurobi(path)
                push!(skipped, path)
                continue
            end
            push!(scripts, path)
        end
    end
    sort!(scripts)
    sort!(skipped)
    for path in skipped
        println("Skipping $(relpath(path, dir)) (requires Gurobi)")
    end

    failures = String[]
    for script in scripts
        rel = relpath(script, dir)
        println("Running $rel")
        cmd = Cmd(
            `$(Base.julia_cmd()) --project=$(abspath(project)) $script`;
            dir=dirname(script),
        )
        success(cmd) || push!(failures, rel)
    end

    if !isempty(failures)
        println("Failed examples:")
        for failed in failures
            println("  ", failed)
        end
        error("$(length(failures)) of $(length(scripts)) examples failed")
    end
    println("Ran $(length(scripts)) examples")
    return scripts
end

# if abspath(PROGRAM_FILE) == @__FILE__
runexamples(; skip_gurobi=get(ENV, "SKIP_GUROBI", "") == "true")
# end
