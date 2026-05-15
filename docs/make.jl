using Documenter
using ButterflyFactorizations
using BEAST # Optional: Include this so the BFBEAST extension is loaded for docs

makedocs(;
    modules=[ButterflyFactorizations],
    authors="Ben Christopher Merten <ben.merten@uni-rostock.de>",
    sitename="ButterflyFactorizations.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Behn98.github.io/ButterflyFactorizations.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        # You can add more pages here later, e.g.:
        # "API Reference" => "api.md"
    ],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/Behn98/ButterflyFactorizations.jl", devbranch="main")
