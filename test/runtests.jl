using PkgSRA, Test

@testset "sindy_turing_test.jl" begin
    include("sindy_turing_test.jl")
end
@testset "sindy_test.jl" begin
    include("sindy_test.jl")
end
@testset "regression_utils_test.jl" begin
    include("regression_utils_test.jl")
end
