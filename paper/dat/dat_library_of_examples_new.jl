using PkgSRA
using Plots, Random, Distributions, Interpolations
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
include("../../utils/sindy_turing_utils.jl")
include("../../utils/sindy_statistics_utils.jl")
include("../../utils/main_algorithm_utils.jl")

include("../../src/sra_model_object.jl");
include("../../src/sra_model_functions.jl");
include("../../src/sra_model_plotting.jl");

################################################################################
#####
##### SIR example: get data
#####
include(EXAMPLE_FOLDERNAME*"example_sir.jl")

# Define the multivariate forcing function
num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0;#0.6;
    amplitude = 100.0
my_U_func_time2(t) = U_func_time(t, u0,
                        U_widths, U_starts,
                        F_dim=1,
                        amplitude=amplitude)

# Get data
sol = solve_sir_system(U_func_time=my_U_func_time2)
dat = Array(sol)
numerical_grad = numerical_derivative(dat, ts)
true_grad = core_dyn_true(dat)

U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t)
end

# Intialize truth object
this_truth = sra_truth_object(true_grad, U_true, core_dyn_true)

#####
##### Build SRA object and analyze
#####
# Initialize
this_model = sra_stateful_object(ts, tspan, dat, u0, numerical_grad)
this_model.parameters.sindyc_ensemble_parameters[:selection_criterion] =
    sindy_cross_validate;
# this_model.parameters.variable_names = ["S", "I", "R"]
fit_first_model(this_model, 100);
print_current_equations(this_model, digits=5)

################################################################
### TESTING
opts = Dict(
    :library=>this_model.parameters.sindy_library,
    :use_lasso=>true,
    :quantile_threshold=>nothing,
    :num_terms=>[1, 2, 1],
    :var_names=>["S", "I", "R"])
m = sindyc(dat, numerical_grad, nothing, ts; opts...)

opts = Dict(
    :library=>this_model.parameters.sindy_library,
    :use_lasso=>false,
    :var_names=>["S", "I", "R"])
m2 = sindyc(dat, numerical_grad, nothing, ts; opts...)

lib = convert_string2function(opts[:library])
aug = calc_augmented_data(dat, lib)
condition_number = LinearAlgebra.cond(aug)
#################################################################

### Iterate
calculate_subsampled_ind(this_model);
all_models = fit_model(this_model);

# println("Iteration $i")
print_current_equations(this_model, digits=5)
print_true_equations(this_truth, digits=5)

plot_subsampled_points(this_model)
# plot_subsampled_simulation(this_model, 2)
plot_subsampled_derivatives(this_model, 1)
plot_residual(this_model)


# Cheating... this works perfectly!
this_model.sindy_model = core_dyn_true;
plot_subsampled_derivatives(this_model, 1)
