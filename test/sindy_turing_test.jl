using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/sindy_turing_utils.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

# Generate chain
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
my_sindy_model = sindyc(dat, numerical_grad,
                        library=sindy_library, use_lasso=true)
my_turing_model = convert_sindy_to_turing_enforce_zeros(my_sindy_model;
                                dat_noise_prior=Normal(0.0, 1.0),
                                coef_noise_std=0.01)

Turing.turnprogress(false)
chain = generate_chain(dat, numerical_grad, my_turing_model,
                            iterations=5,
                            num_training_pts=100, start_ind=101)[1]

# Number of terms tests
terms = get_nonzero_terms(my_sindy_model)
@test length(chain.name_map.parameters) - 1 == length(terms)
@test chain.name_map.parameters[end] == "noise"

# Converting back to SINDy model
my_sindy_sample = sindy_from_chain(my_sindy_model,
                                   chain,
                                   enforced_zeros=true)
terms_sample = get_nonzero_terms(my_sindy_sample)
@test terms == terms_sample

# Test helper prediction function
pred1 = my_sindy_model(dat[:,1], 0)
pred2 = sindy_predict(my_sindy_model,
                      my_sindy_model.A[terms],
                      dat[:,1],
                      nonzero_terms=terms)
@test all(pred1 .== pred2)
