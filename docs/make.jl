using Documenter
using ButterflyFactorizations

makedocs(;
    sitename="ButterflyFactorizations.jl",
    modules=[ButterflyFactorizations],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", assets=String[]),
    pages=[
        "Home & API" => "index.md",
        "Mathematical Theory" => "theory.md",
        "Performance Benchmarks" => "benchmarks.md",
    ],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/Behn98/ButterflyFactorizations.jl.git", devbranch="main")
