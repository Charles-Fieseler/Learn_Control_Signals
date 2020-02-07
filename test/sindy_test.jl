using PkgSRA, Test, Random
Random.seed!(13)

# include("../utils/sindy_turing_utils.jl")
# include("../utils/sparse_regression_functions.jl")
# include("../utils/sindy_statistics_utils.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

# Test new interface
alg = slstQuantile(4, 0.2) # Will get extra terms
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
test_model = sindy(dat, numerical_grad, ts,
                        library=sindy_library,
                        optimizer=alg)

# Built correct terms: names
named_terms = build_term_names(test_model)
@testset "Term Metadata" begin
    @test named_terms ==
        ["x", "y", "z", "", "xx", "xy", "xz", "yy", "yz", "zz"]
end

# Built correct terms: values
library_functions = convert_string2function(test_model.library)
terms = calc_augmented_data(dat, library_functions)


@testset "Data Augmentation" begin
    @test size(terms,1) == length(named_terms)
    @test terms[1:3,:] == dat
    @test terms[4,:] == ones(size(dat,2))
    @test terms[5:10,:] == calc_cross_terms(dat, 2)
end;

# Nonzero term and type tests
t1 = get_nonzero_terms(test_model)
t2 = get_nonzero_terms(core_dyn_true)
# typeof(test_model) <: sindycModel
# @testset "Type Tests" begin
@test typeof(test_model) <: sindyModel
@test issubset(t2, t1)
# end

###
### Cross validation functions
###

# First, retraining functions
test_model2 = sindy_retrain(test_model, dat, numerical_grad)
t1 = get_nonzero_terms(test_model)
t2 = get_nonzero_terms(test_model2)

@testset "Retrained model" begin
    @test typeof(test_model2) <: sindyModel
    @test t1==t2
    @test calc_coefficient_error(test_model, test_model2) ≈ 0.0
end

# Also test cross validation function
null_model = sindy(dat, numerical_grad, ts,
                        library=sindy_library,
                        optimizer=slstHard(100)) # i.e. not sparse
err_test = sindy_cross_validate(test_model, dat, numerical_grad)
err_null = sindy_cross_validate(null_model, dat, numerical_grad)

@testset "Cross Validation" begin
    @test err_test < err_null
end


## One more cross validation test
include("../examples/example_lotkaVolterra.jl")
dat = Array(solve_lv_system())
numerical_grad = numerical_derivative(dat, ts)

# First, a decent model on one variable
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
lib = convert_string2function(sindy_library)
#      x   y   c   xx xy yy
A1 = [[0.8 0   0   0 -1.4 0 ];
      [0  0   0   0   0   1e-6 ]]
n = size(A, 1)
m1 = sindyModel(ts, A1, lib ,["x", "y"])

# Second, a decent model on both variables
#      x   y   c   xx xy yy
A2 = [[0.7 0   0   0 -1.3 1e-6 ];
      [0  -0.7 0  1e-6 0.7  0]]
m2 = sindyModel(ts, A2, lib ,["x", "y"])

## Calculate errors
# Coefficients
cerr1 = calc_coefficient_error(m1, core_dyn_true)
cerr2 = calc_coefficient_error(m2, core_dyn_true)
# Residual
ind = 1:500
rss0 = rss_sindy_derivs(core_dyn_true, dat[:,ind], numerical_grad[:,ind])
rss1 = rss_sindy_derivs(m1, dat[:,ind], numerical_grad[:,ind])
rss2 = rss_sindy_derivs(m2, dat[:,ind], numerical_grad[:,ind])

#
@testset "Error calculations" begin
    @test cerr1 > cerr2
    @test rss1 > rss0
    @test rss2 > rss0
    @test rss1 > rss2
end

using Plots
pyplot()
u0 = dat[:,1]
plot_sindy_model(m1, u0, which_dim=2)
    plot!(m1.ts, dat[2,:])
    title!("Model 1")
plot_sindy_model(m2, u0, which_dim=2)
    plot!(m2.ts, dat[2,:])
    title!("Model 2")



# SINDY tests: up to cubed terms

# Generate test data: Van der Pol
include("../examples/example_vanDerPol.jl")
dat = Array(solve_vdp_system())
numerical_grad = numerical_derivative(dat, ts)


sindy_library = Dict("cross_terms"=>[2,3],
                    "constant"=>nothing);
test_model = sindy(dat, numerical_grad, ts,
                        library=sindy_library,
                        optimizer=alg,
                        var_names=["x", "y"])

named_terms = build_term_names(test_model)
@testset "Cubic term names" begin
    @test named_terms ==
        ["x", "y", "", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy"]
end
