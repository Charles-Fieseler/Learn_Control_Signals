using PkgSRA
using Plots, Random, Distributions, Interpolations
using StatsPlots
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
# include("../../utils/sindy_turing_utils.jl")
# include("../../utils/main_algorithm_utils.jl")
# include("../../utils/sindy_statistics_utils.jl")

################################################################################
#####
##### Lotka-Volterra example: get data
#####
include(EXAMPLE_FOLDERNAME*"example_lotkaVolterra.jl")

# Define the multivariate forcing function
num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0.3;
    # U_widths = 0;
    amplitude = 1.0
my_U_func_time2(t) = U_func_time(t, u0,
                        U_widths, U_starts,
                        F_dim=1,
                        amplitude=amplitude)

# Get data
sol = solve_lv_system(U_func_time=my_U_func_time2)
dat = Array(sol)
numerical_grad = numerical_derivative(dat, ts)
true_grad = core_dyn_true(dat)

U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t)
end

## Also get baseline true/ideal cases
# Uncontrolled
dat_raw = Array(solve_lv_system())
numerical_grad_raw = numerical_derivative(dat_raw, ts)
true_grad_raw = zeros(size(dat))

# Intialize truth object
this_truth = sra_truth_object(dat_raw, true_grad, U_true, core_dyn_true)

#####
##### Loop over additive noise
#####

noise_vals = [0, 0]
# noise_factor = norm(numerical_grad)
# noise_factor = 1;
# noise_vals .*= noise_factor
# noise_factor = sqrt.(sum(numerical_grad.^2, dims=2))
num_models = 5
all_err = zeros(length(noise_vals), num_models)


for (i,σ) in enumerate(noise_vals)
    for j in 1:num_models
        noisy_grad = numerical_grad .+ σ.*randn(size(dat))
        # Initialize
        this_model = sra_stateful_object(ts, tspan, dat, u0, noisy_grad)
        prams = this_model.parameters
        # Reset parameters because we are NOT using control
        # prams.sindyc_ensemble_parameters =
        #     Dict(:selection_criterion=>sindy_cross_validate);
        # prams.sindyc_ensemble_parameters[:selection_criterion] =
        #     sindy_cross_validate;
        prams.model_template.variable_names = ["x", "y"];
        prams.model_template.sindy_library[calc_cross_terms] = [2] # Also include cubic terms
        prams.sindy_terms_list = Iterators.product(1:3, 1:3)
        prams.initial_subsampling = true

        fit_first_model(this_model, 30);
        print_true_equations(this_truth)
        print_current_equations(this_model)

        #################################################################
        ### Iterate
        is_improved, max_iter = true, 10
        while is_improved
            calculate_subsampled_ind(this_model);
            all_models, is_improved = fit_model(this_model);
            is_improved && print_current_equations(this_model)
            this_model.i > 10 && break
        end
        println("Finished with $(this_model.i) iterations")

        all_err[i, j] = calc_coefficient_error(this_model, this_truth)
        println("Final error = $(all_err[i, j])")
        println("")
    end
    println("Finished noise level $σ")
    println("=====================================")
end

# boxplot(noise_vals, all_err')

vec_err, std_err = mean_and_std(all_err, 2)
# vec_err, std_err = mean_and_std(all_err)
plot(noise_vals, vec_err, ribbon=std_err)
    xlabel!("Noise")
    ylabel!("Coefficient Error")

# plot_subsampled_points(this_model)
# plot_subsampled_simulation(this_model, 2)
# plot_subsampled_derivatives(this_model, 2)
# plot_residual(this_model)

# plot_subsampled_points_and_control(this_model, this_truth)

#####
##### Save Lotka-Volterra data
#####
this_dat_name = "dat_library_of_examples_lv_noise_"

save_for_plotting(this_model, this_truth, this_dat_name)
