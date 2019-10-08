using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/sindy_turing_utils.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

# SINDY tests
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
test_model = sindyc(dat, numerical_grad,
                        library=sindy_library, use_lasso=true)

# Type tests
@test typeof(test_model) <: sindyc_model

# Nonzero term tests
t1 = get_nonzero_terms(test_model)
t2 = get_nonzero_terms(core_dyn_true)

@test issubset(t2, t1)
