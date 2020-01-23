using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/sindy_turing_utils.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

# SINDY tests: up to squared terms
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
test_model = sindyc(dat, numerical_grad, nothing, ts,
                        library=sindy_library,
                        use_lasso=true)

# Built correct terms: names
named_terms = build_term_names(test_model)
@test named_terms ==
    ["x", "y", "z", "", "xx", "xy", "xz", "yy", "yz", "zz"]

# Built correct terms: values
library_functions = convert_string2function(test_model.library)
terms = calc_augmented_data(dat, library_functions)

@test size(terms,1) == length(named_terms)

@testset "Data Augmentation" begin
    @test terms[1:3,:] == dat
    @test terms[4,:] == ones(size(dat,2))
    @test terms[5:10,:] == calc_cross_terms(dat, 2)
end;

# Type tests
@test typeof(test_model) <: sindyc_model

# Nonzero term tests
t1 = get_nonzero_terms(test_model)
t2 = get_nonzero_terms(core_dyn_true)

@test issubset(t2, t1)



# SINDY tests: up to cubed terms

# Generate test data: Van der Pol
include("../examples/example_vanDerPol.jl")
dat = Array(solve_vdp_system())
numerical_grad = numerical_derivative(dat, ts)


sindy_library = Dict("cross_terms"=>[2,3],
                    "constant"=>nothing);
test_model2 = sindyc(dat, numerical_grad, nothing, ts,
                        library=sindy_library,
                        use_lasso=true,
                        var_names=["x", "y"])

named_terms = build_term_names(test_model2)
@test named_terms ==
    ["x", "y", "", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy"]
