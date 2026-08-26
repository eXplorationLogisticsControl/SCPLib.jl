"""Test non-convex gradient cache callbacks."""

using Test

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

@testset "non-convex gradients receive linearization cache" begin
    caches = (
        SCPLib.MultipleShootingCache(1, 1, 2, 1, 1, 1),
        SCPLib.ForwardBackwardCache(1, 1, 2, 1, 1, 1),
    )

    for cache in caches
        x_ref = zeros(1, 2)
        u_ref = zeros(1, 1)

        g_called_with_cache = Ref(false)
        function ∇g_noncvx(cache_arg, x, u)
            g_called_with_cache[] = cache_arg === cache
            return fill(2.0, 1, size(cache.∇g, 2))
        end

        h_called_with_cache = Ref(false)
        function ∇h_noncvx(cache_arg, x, u)
            h_called_with_cache[] = cache_arg === cache
            return fill(3.0, 1, size(cache.∇h, 2))
        end

        SCPLib.set_g_noncvx_cache!(cache, ∇g_noncvx, x_ref, u_ref)
        SCPLib.set_h_noncvx_cache!(cache, ∇h_noncvx, x_ref, u_ref)

        @test g_called_with_cache[]
        @test h_called_with_cache[]
        @test cache.∇g == fill(2.0, 1, size(cache.∇g, 2))
        @test cache.∇h == fill(3.0, 1, size(cache.∇h, 2))
    end
end

@testset "non-convex gradients support legacy signature" begin
    cache = SCPLib.MultipleShootingCache(1, 1, 2, 1, 1, 1)
    x_ref = zeros(1, 2)
    u_ref = zeros(1, 1)

    SCPLib.set_g_noncvx_cache!(cache, (x, u) -> fill(4.0, 1, size(cache.∇g, 2)), x_ref, u_ref)
    SCPLib.set_h_noncvx_cache!(cache, (x, u) -> fill(5.0, 1, size(cache.∇h, 2)), x_ref, u_ref)

    @test cache.∇g == fill(4.0, 1, size(cache.∇g, 2))
    @test cache.∇h == fill(5.0, 1, size(cache.∇h, 2))
end
