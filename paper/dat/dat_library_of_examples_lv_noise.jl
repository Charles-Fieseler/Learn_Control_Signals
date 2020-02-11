using PkgSRA
using Plots, Random, Distributions, Interpolations
using StatsPlots, StatsBase
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
num_ctr = 7;
    U_starts = rand(1, num_ctr) .* tspan[2]
    U_widths = 0.4;
    # U_widths = 0;
    amplitude = 2.0
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

plot(sol)
    plot!(ts, U_true[1,:])
    title!("Data")

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

# noise_vals = [0, 0.1]
noise_vals = 0.0:0.05:0.2
# noise_factor = norm(numerical_grad)
# noise_factor = 1;
# noise_vals .*= noise_factor
# noise_factor = sqrt.(sum(numerical_grad.^2, dims=2))
num_models = 10
all_err = zeros(length(noise_vals), num_models)
all_err_deriv = zeros(length(noise_vals), num_models)
all_naive_err = zeros(length(noise_vals), num_models)
all_naive_err_deriv = zeros(length(noise_vals), num_models)
all_i = zeros(length(noise_vals), num_models)

# Make model template
variable_names = ["x", "y"]
optimizer = slstNumber((1, 1));
library = Dict("cross_terms"=>[2], "constant"=>nothing)
model_template = sindyModel(convert_string2function(library),
    variable_names, optimizer)

for (i,σ) in enumerate(noise_vals)
    for j in 1:num_models
        noisy_grad = numerical_grad .+ σ.*randn(size(dat))
        # Initialize
        global this_model = sra_stateful_object(ts, tspan, dat, u0, noisy_grad)
        # Reset parameters because we are NOT using control
        this_model.parameters = get_sra_defaults(false)
        prams = this_model.parameters
        prams.model_template = model_template
        # prams.sindyc_ensemble_parameters =
        #     Dict(:selection_criterion=>sindy_cross_validate);
        # prams.sindyc_ensemble_parameters[:selection_criterion] =
        #     sindy_cross_validate;
        prams.sindy_terms_list = Iterators.product(1:3, 1:3)
        prams.initial_subsampling = true

        fit_first_model(this_model, 30);
        print_true_equations(this_truth)
        print_current_equations(this_model)
        all_naive_err[i, j] = calc_coefficient_error(this_model, this_truth)
        all_naive_err_deriv[i, j] = rss_sindy_derivs(
                this_model.sindy_model, dat, numerical_grad)

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
        all_err_deriv[i, j] = rss_sindy_derivs(
                this_model.sindy_model, dat, numerical_grad)
        all_i[i, j] = this_model.i
        println("Final error in coefficients = $(all_err[i, j])")
        println("")
    end
    println("Finished noise level $σ")
    println("=====================================")
end

# all_err_saved = copy(all_err);
# boxplot(collect(noise_vals)', all_err')

# all_err = all_err_saved;
# Remove outliers
# for i in 1:size(all_err,1)
#     all_err[i,:] = replace_outliers(all_err[i,:], 0)
# end
coef_norm = sum(core_dyn_true.A.^2)
vec_err, std_err = mean_and_std(all_err./coef_norm, 2)
# vec_err, std_err = mean_and_std(all_err)
plot(noise_vals, vec_err, ribbon=std_err)
    xlabel!("Noise")
    ylabel!("Coefficient Error")

# vec_i, std_i = mean_and_std(all_i, 2)
# # vec_err, std_err = mean_and_std(all_err)
# plot(noise_vals, vec_i, ribbon=std_err)
#     xlabel!("Noise")
#     ylabel!("Number of iterations")
#     title!("Average number of iterations to convergence")

vec_naive = mean(all_naive_err, dims=2)
vec_diff, std_diff = mean_and_std((vec_naive .- all_err)./coef_norm, 2)
plot(noise_vals, vec_diff, ribbon=std_diff)
    xlabel!("Noise")
    ylabel!("Average improvement")
    title!("Percentage coefficient improvement")

# Error in derivatives: only look at where there's no control
# ind = this_model.subsample_ind
vec_naive_deriv = mean(all_naive_err_deriv, dims=2)
vec_deriv = mean(all_err_deriv, dims=2)
vec_diff_deriv, std_diff_deriv = mean_and_std(vec_naive_deriv .- vec_deriv, 2)
plot(noise_vals, vec_diff_deriv, ribbon=std_diff_deriv)
    xlabel!("Noise")
    ylabel!("Average improvement")
    title!("Error in derivatives")

# Model visualizations
plot_subsampled_points(this_model)
# plot_subsampled_simulation(this_model, 2)
plot_subsampled_derivatives(this_model, 1)
plot_residual(this_model)

# plot_subsampled_points_and_control(this_model, this_truth)

#####
##### Save Lotka-Volterra data
#####
this_dat_name = "dat_library_of_examples_lv_noise_"

save_for_plotting(this_model, this_truth, this_dat_name)
