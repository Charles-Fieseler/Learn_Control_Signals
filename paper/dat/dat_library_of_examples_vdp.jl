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
include(EXAMPLE_FOLDERNAME*"example_vanDerPol.jl")

# Define the multivariate forcing function
num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0.6;
    amplitude = 100.0
my_U_func_time2(t) = U_func_time(t, u0,
                        U_widths, U_starts,
                        F_dim=1,
                        amplitude=amplitude)

# Get data
sol = solve_vdp_system(U_func_time=my_U_func_time2)
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
p = this_model.parameters
p.sindyc_ensemble_parameters[:selection_criterion] =
    sindy_cross_validate;
# Several changes are required because there are only 2 variables here
# p.sindyc_ensemble_parameters[:variable_names] = ["x", "y"];
p.variable_names = ["x", "y"];
p.sindy_terms_list = Iterators.product(1:3, 1:3)
p.sindy_library["cross_terms"] = [2, 3] # Also include cubic terms

fit_first_model(this_model, 100);
print_current_equations(this_model)

#################################################################
### Iterate
calculate_subsampled_ind(this_model);
all_models = fit_model(this_model);

# println("Iteration $i")
print_current_equations(this_model)
print_true_equations(this_truth)

plot_subsampled_points(this_model)
# plot_subsampled_simulation(this_model, 2)
plot_subsampled_derivatives(this_model, 1)
plot_residual(this_model)
