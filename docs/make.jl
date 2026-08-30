using Documenter
using ButterflyFactorizations

makedocs(;
    sitename="ButterflyFactorizations.jl",
    modules=[ButterflyFactorizations],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        logo="assets/logo.png", # Enables your new transparent logo in the sidebar
        assets=String[],
    ),
    pages=[
        "Home & API" => "index.md",
        "Mathematical Theory" => "theory.md",
        "Performance Benchmarks" => "benchmarks.md",
    ],
)

deploydocs(; repo="github.com/Behn98/ButterflyFactorizations.jl.git", devbranch="main")
