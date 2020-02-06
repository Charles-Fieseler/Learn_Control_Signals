using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/regression_utils.jl")
include("../utils/sparse_regression_functions.jl")

# Generate test data: Lorenz
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
numerical_grad = numerical_derivative(dat, ts)

sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);

make_model(x) = sindyc(slstNumber(2, x), dat, numerical_grad)

#####
##### Test Sparse Regression function
#####
thresh = 0.1
m1 = slstHard(2, thresh)
A = sparse_regression(m1, dat, numerical_grad)
@testset "Hard threshold" begin
    @test all(abs.(A).<thresh)
end

#####
##### Test sparse models: all equations same terms
#####
num_terms5 = [5]
terms5 = get_nonzero_terms(make_model(num_terms5))
num_terms3 = [3]
terms3 = get_nonzero_terms(make_model(num_terms3))
@testset "Number of terms" begin
    @test length(terms5) == size(dat,1)*num_terms5
    @test length(terms3) == size(dat,1)*num_terms5
end

#####
##### Test sparse models: all equations different terms
#####
@testset "Number of terms2" begin
    num_terms = [1, 2, 3]
    terms = get_nonzero_terms(make_model(num_terms))
    @test length(terms) == sum(num_terms)

    num_terms = [2, 3, 4]
    terms = get_nonzero_terms(make_model(num_terms))
    @test length(terms) == sum(num_terms)
end

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

@testset "Ensemble models" begin
    i = 1
    @test length(get_nonzero_terms(all_models[i])) == size(dat,1)*val_list[i]
    i = 2
    @test length(get_nonzero_terms(all_models[i])) == size(dat,1)*val_list[i]
end

# Permutations
# val_list = calc_permutations(5,3)
# val_list = combinations(1:5, 3)
val_list = Iterators.product(1:3,1:3,1:3) # DEBUG
(best_model,best_criterion,all_criteria,all_models) =
    make_ensemble(val_list)

@testset "Permutations" begin
    for (i, v) in enumerate(val_list)
        @test length(get_nonzero_terms(all_models[i])) == sum(v)
    end
end

# The best model should be almost the real one
@testset "Accuracy" begin
    for (real_val, model_val) in zip(vec(core_dyn_true.A), vec(best_model.A))
        @test isapprox(Float64(real_val), model_val, atol=1e-1)
    end
end
