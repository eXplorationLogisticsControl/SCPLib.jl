"""Utility functions"""


function vprintf(verbosity::Int, str::String)
    if verbosity > 0
        println(str)
    end
end


function message_accept_step(accept::Bool)
    if accept
        return "yes"
    else
        return "no "
    end
end