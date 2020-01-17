using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/regression_utils.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);

make_model(x) = sindyc(dat, numerical_grad,
                    library=sindy_library, use_lasso=true,
                    quantile_threshold=nothing,
                    num_terms=x)

#####
##### Test sparse models: all equations same terms
#####
num_terms = 5
terms = get_nonzero_terms(make_model(num_terms))
@test length(terms) == size(dat,1)*num_terms

#
num_terms = 3
terms = get_nonzero_terms(make_model(num_terms))
@test length(terms) == size(dat,1)*num_terms

#####
##### Test sparse models: all equations different terms
#####
num_terms = [1, 2, 3]
terms = get_nonzero_terms(make_model(num_terms))
@test length(terms) == sum(num_terms)

num_terms = [2, 3, 4]
terms = get_nonzero_terms(make_model(num_terms))
@test length(terms) == sum(num_terms)


#####
##### Test ensemble models
#####
make_ensemble(x) = sindyc_ensemble(dat, numerical_grad,
                    sindy_library,
                    x,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms")

# Each ensemble has all same
val_list = [1, 2]
(best_model_raw,best_criterion,all_criteria,all_models) =
    make_ensemble(val_list)

i = 1
@test length(get_nonzero_terms(all_models[i])) == size(dat,1)*val_list[i]
i = 2
@test length(get_nonzero_terms(all_models[i])) == size(dat,1)*val_list[i]

# Permutations
# val_list = calc_permutations(5,3)
# val_list = combinations(1:5, 3)
val_list = Iterators.product(1:3,1:3,1:3) # DEBUG
(best_model,best_criterion,all_criteria,all_models) =
    make_ensemble(val_list)

for (i, v) in enumerate(val_list)
    @test length(get_nonzero_terms(all_models[i])) == sum(v)
end

# The best model should be almost the real one
for (real_val, model_val) in zip(vec(core_dyn_true.A), vec(best_model.A))
    @test isapprox(Float64(real_val), model_val, atol=1e-1)
end
