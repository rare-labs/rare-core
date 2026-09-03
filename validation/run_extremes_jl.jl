# Fit committed golden samples with Extremes.jl and write JSON snapshots.
# Pinned commit: fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc
# Not used by normal CI. Install Extremes.jl at that revision, then:
#   julia validation/run_extremes_jl.jl
#
# Extremes.jl MLE entry points are gevfit / gpfit. params(fm)[1] is (μ, σ, ξ)
# with Coles ξ (same sign as this engine). location/scale/shape return 1-vectors.

using JSON
using Extremes

const ROOT = @__DIR__
const SAMPLES = joinpath(ROOT, "fixtures", "samples")
const CATALOG = joinpath(ROOT, "fixtures", "catalog.json")
const OUT = joinpath(ROOT, "fixtures", "references", "extremes_jl")
const PERIODS = (2.0, 10.0, 50.0)

function gev_return_level(μ, σ, ξ, T)
    p = 1 - 1 / T
    if abs(ξ) < 1e-8
        return μ - σ * log(-log(p))
    end
    return μ + (σ / ξ) * ((-log(p))^(-ξ) - 1)
end

function gpd_return_level(u, σ, ξ, λ, T)
    m = λ * T
    if abs(ξ) < 1e-8
        return u + σ * log(m)
    end
    return u + (σ / ξ) * (m^ξ - 1)
end

function write_json(path, payload)
    open(path, "w") do io
        JSON.print(io, payload, 2)
        println(io)
    end
end

function main()
    mkpath(OUT)
    catalog = JSON.parsefile(CATALOG)
    for entry in catalog["fixtures"]
        name = entry["name"]
        model = entry["model"]
        sample = JSON.parsefile(joinpath(SAMPLES, name * ".json"))
        if model == "GEV"
            y = Float64.(sample["transformed_values"])
            fm = gevfit(y)
            μ, σ, ξ = params(fm)[1]
            ll = loglike(fm)
            fit = Dict(
                "schema_version" => "0.1",
                "type" => "GEVFit",
                "location" => μ,
                "scale" => σ,
                "shape" => ξ,
                "log_likelihood" => ll,
                "aic" => 6.0 - 2.0 * ll,
                "n_extremes" => length(y),
                "converged" => true,
            )
            levels = Dict(string(Int(T)) => gev_return_level(μ, σ, ξ, T) for T in PERIODS)
        else
            y = Float64.(sample["excesses"])
            fm = gpfit(y)
            _, σ, ξ = params(fm)[1]
            ll = loglike(fm)
            fit = Dict(
                "schema_version" => "0.1",
                "type" => "GPDFit",
                "scale" => σ,
                "shape" => ξ,
                "log_likelihood" => ll,
                "aic" => 4.0 - 2.0 * ll,
                "n_extremes" => length(y),
                "converged" => true,
            )
            u = Float64(sample["threshold"])
            λ = Float64(sample["exceedance_rate"])
            levels = Dict(string(Int(T)) => gpd_return_level(u, σ, ξ, λ, T) for T in PERIODS)
        end
        payload = Dict(
            "model" => model,
            "fit" => fit,
            "return_levels" => levels,
            "log_likelihood" => fit["log_likelihood"],
        )
        out_path = joinpath(OUT, name * ".json")
        write_json(out_path, payload)
        println("wrote ", out_path)
    end
end

main()
