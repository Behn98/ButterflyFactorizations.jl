using Test, TestItems, TestItemRunner

@testitem "ButterflyFactorizations" begin end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(ButterflyFactorizations; unbound_args=false)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter, ButterflyFactorizations
    @test JuliaFormatter.format(pkgdir(ButterflyFactorizations), overwrite=false)
end

@run_package_tests verbose = true
