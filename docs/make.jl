using ButterflyFactorizations
using Documenter

DocMeta.setdocmeta!(
    ButterflyFactorizations, :DocTestSetup, :(using ButterflyFactorizations); recursive=true
)

makedocs(;
    modules=[ButterflyFactorizations],
    authors="Ben Christopher Merten <ben.merten@uni-rostock.de>",
    sitename="ButterflyFactorizations.jl",
    format=Documenter.HTML(;
        canonical="https://Behn98.github.io/ButterflyFactorizations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/Behn98/ButterflyFactorizations.jl", devbranch="main")
