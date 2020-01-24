using PkgSRA, Test, Random
Random.seed!(13)

include("../utils/sindy_turing_utils.jl")

# Generate test data: Van der Pol (2-d)
include("../examples/example_vanDerPol.jl")
dat = Array(solve_vdp_system())
numerical_grad = numerical_derivative(dat, ts)

# Initialize main SRA object
this_model = sra_stateful_object(ts, tspan, dat, u0, numerical_grad)
prams = this_model.parameters
prams.sindyc_ensemble_parameters[:selection_criterion] =
    sindy_cross_validate;
# Several changes are required because there are only 2 variables here
prams.variable_names = ["x", "y"];
prams.sindy_terms_list = Iterators.product(1:3, 1:3)
prams.sindy_library["cross_terms"] = [2, 3] # Also include cubic terms

# Fit; iteration not necessary here
fit_first_model(this_model, 1);
calculate_subsampled_ind(this_model);
all_models = fit_model(this_model);

@testset "Exact Nonzero terms" begin
    names = get_nonzero_term_names(all_models[1]);
    @test names == ["dxdt_x", "dydt_x"]
    names = get_nonzero_term_names(all_models[2]);
    @test names == ["dxdt_x", "dydt_x", "dxdt_y"]
    names = get_nonzero_term_names(all_models[3]);
    @test names == ["dxdt_x", "dydt_x", "dxdt_y", "dxdt_xxx"]
    # TODO: finish
end;

@testset "Correct Regression" begin
    names = get_nonzero_term_names(this_model.sindy_model);
    @test names == ["dxdt_x",
                    "dydt_x",
                    "dxdt_y",
                    "dydt_y", # Spurious term
                    "dydt_", # Spurious term
                    "dxdt_xxx"]
end;
