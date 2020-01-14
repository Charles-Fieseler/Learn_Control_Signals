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
    U_widths = 0.6;
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
fit_first_model(this_model, 100);


### Iterate
calculate_subsampled_ind(this_model);
fit_model(this_model);

# println("Iteration $i")
print_current_equations(this_model, digits=5)
print_true_equations(this_truth, digits=5)

plot_subsampled_points(this_model)
plot_subsampled_simulation(this_model)
plot_residual(this_model)
