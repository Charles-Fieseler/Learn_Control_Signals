using PkgSRA
using Plots, Random, Distributions, Interpolations
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
include("../../utils/sindy_turing_utils.jl")
include("../../utils/main_algorithm_utils.jl")
include("../../utils/sindy_statistics_utils.jl")

################################################################################
#####
##### SIR example: get data
#####
include(EXAMPLE_FOLDERNAME*"example_fitzHughNagumo.jl")

# Define the multivariate forcing function
num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0;0.6;
    amplitude = 1.0
my_U_func_time2(t) = U_func_time(t, u0,
                        U_widths, U_starts,
                        F_dim=1,
                        amplitude=amplitude)

# Get data
sol = solve_fhn_system(U_func_time=my_U_func_time2)
dat = Array(sol)
numerical_grad = numerical_derivative(dat, ts)
true_grad = core_dyn_true(dat)

U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t)
end

## Also get baseline true/ideal cases
# Uncontrolled
dat_raw = Array(solve_fhn_system())
numerical_grad_raw = numerical_derivative(dat_raw, ts)
true_grad_raw = zeros(size(dat))

# Intialize truth object
this_truth = sra_truth_object(dat_raw, true_grad, U_true, core_dyn_true)

#####
##### Build SRA object and analyze
#####
# Initialize
this_model = sra_stateful_object(ts, tspan, dat, u0, numerical_grad)
prams = this_model.parameters
prams.sindyc_ensemble_parameters[:selection_criterion] =
    sindy_cross_validate;
# Several changes are required because there are only 2 variables here
prams.variable_names = ["v", "ω"];
prams.sindy_terms_list = Iterators.product(1:4, 1:4)
prams.sindy_library["cross_terms"] = [2, 3] # Also include cubic terms

fit_first_model(this_model, 1);
print_true_equations(this_truth)
print_current_equations(this_model)

plot_sindy_model(this_model, 1)

#################################################################
### Iterate
calculate_subsampled_ind(this_model);
all_models = fit_model(this_model);

print_true_equations(this_truth)
print_current_equations(this_model)

# plot_subsampled_points(this_model)
plot_subsampled_simulation(this_model, 1)
plot_subsampled_derivatives(this_model, 1)
# plot_residual(this_model)

# this_model.U_true

# plot_subsampled_points_and_control(this_model, this_truth)


#####
##### Save FitzHugh data
#####
this_dat_name = "dat_library_of_examples_fhn_"

save_for_plotting(this_model, this_truth, this_dat_name)
