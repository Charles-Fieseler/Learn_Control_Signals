using PkgSRA
using Plots, Random, Distributions, Interpolations
using StatsPlots, StatsBase
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");

################################################################################
#####
##### Lotka-Volterra example: get data
#####
include(EXAMPLE_FOLDERNAME*"example_lotkaVolterra.jl")

# Define the multivariate forcing function
function local_make_data(num_ctr)
    # num_ctr = 7;
    U_starts = rand(1, num_ctr) .* tspan[2]
    U_widths = 0.4;
    amplitude = 2.0
    my_U_func_time2(t) = U_func_time(t, u0,
                            U_widths, U_starts,
                            F_dim=1,
                            amplitude=amplitude)
    # Get data
    sol = solve_lv_system(U_func_time=my_U_func_time2)
    dat = Array(sol)

    # Also get truth
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
    #
    # # Intialize truth object
    this_truth = sra_truth_object(dat_raw, true_grad, U_true, core_dyn_true)

    # End
    return this_truth, dat
end

#####
##### Loop over noise and number of spikes
#####
noise_vals = 0.0:0.05:0.5
num_models = 50
control_signal_vals = 0:3:12
# noise_vals = 0.0:0.05:0.2
# num_models = 10
# control_signal_vals = [0, 5, 10]

## Output values
this_size = (length(noise_vals), length(control_signal_vals), num_models)
all_err = zeros(this_size)
# all_err_deriv = zeros(this_size)
all_err_deriv_subsample = zeros(this_size)
all_naive_err = zeros(this_size)
all_naive_err_deriv = zeros(this_size)

# all_i = zeros(this_size)

# Make model template
function local_param_reset!(this_model)
    variable_names = ["x", "y"]
    optimizer = slstNumber((1, 1)); # Number of terms-based sparse regression
    library = Dict("cross_terms"=>[2], "constant"=>nothing)
    model_template = sindyModel(convert_string2function(library),
        variable_names, optimizer)
    this_model.parameters = get_sra_defaults(false)
    prams = this_model.parameters
    prams.model_template = model_template
    prams.sindy_terms_list = Iterators.product(1:3, 1:3)
    prams.initial_subsampling = true
end

for (i_noise,σ) in enumerate(noise_vals)
    for (i_ctr, num_ctr) in enumerate(control_signal_vals)
    this_truth, dat = local_make_data(num_ctr)
        for i_model in 1:num_models
            # Add noise, THEN take derivatives
            noisy_dat = dat .+ σ.*randn(size(dat))
            noisy_grad = numerical_derivative(noisy_dat, ts)
            # Initialize
            global this_model =
                sra_stateful_object(ts, tspan, dat, u0, noisy_grad)
            # Reset parameters because we are NOT using sindyc
            local_param_reset!(this_model)

            fit_first_model(this_model, 30);
            # print_true_equations(this_truth)
            # print_current_equations(this_model)
            all_naive_err[i_noise, i_ctr, i_model] =
                    calc_coefficient_error(this_model, this_truth)
            all_naive_err_deriv[i_noise, i_ctr, i_model] = rss_sindy_derivs(
                    this_model.sindy_model, noisy_dat, this_truth.true_grad) ./
                    length(dat)

            #################################################################
            ### Iterate
            is_improved, max_iter = true, 10
            while is_improved
                calculate_subsampled_ind(this_model);
                all_models, is_improved = fit_model(this_model);
                # is_improved && print_current_equations(this_model)
                this_model.i > max_iter && break
            end
            println("Finished with $(this_model.i) iterations")

            all_err[i_noise, i_ctr, i_model] =
                calc_coefficient_error(this_model, this_truth)
            # all_err_deriv[i_noise, i_ctr, i_model] = rss_sindy_derivs(
            #         this_model.sindy_model, dat, numerical_grad)
            ind = this_model.subsample_ind
            all_err_deriv_subsample[i_noise, i_ctr, i_model] = rss_sindy_derivs(
                    this_model.sindy_model, noisy_dat[:,ind], this_truth.true_grad[:,ind]) ./
                    (length(ind)*size(dat,1))
            # all_i[i_noise, i_ctr, i_model] = this_model.i
            # println("Final error in coefficients = $(all_err[i_noise, i_model])")
            # println("")
        end
        println("Finished number of control signals $num_ctr")
    end
    println("Finished noise level $σ")
    println("=====================================")
end

coef_norm = sum(core_dyn_true.A.^2)

# heatmap_coef_naive = mean(all_naive_err./coef_norm, dims=3)[:,:,1]
# heatmap_coef_final = mean(all_err./coef_norm, dims=3)[:,:,1]
# heatmap(control_signal_vals, noise_vals, heatmap_coef_final)
#     xlabel!("Controllers")
#     ylabel!("Noise")
#     title!("Final Model Error")
# heatmap(control_signal_vals, noise_vals, heatmap_coef_naive .- heatmap_coef_final)
#     xlabel!("Controllers")
#     ylabel!("Noise")
#     title!("Improvement in error")

# vec_naive, std_naive = mean_and_std(all_naive_err./coef_norm, 2)
# vec_err, std_err = mean_and_std(all_err./coef_norm, 2)
# # Error in derivatives: only look at where there's no control
# vec_deriv, std_deriv = mean_and_std(all_err_deriv_subsample, 2)
# vec_naive_deriv, std_naive_deriv = mean_and_std(all_naive_err_deriv, 2)

#####
##### Save Lotka-Volterra data
#####
this_dat_name = "dat_library_with_noise_lv_"

# save_for_plotting(noise_vals,
#                 vec_naive, std_naive, vec_err, std_err,
#                 vec_deriv, std_deriv, vec_naive_deriv, std_naive_deriv,
#                 this_dat_name)

save_for_plotting(noise_vals, control_signal_vals, coef_norm,
                all_naive_err, all_err,
                all_err_deriv_subsample, all_naive_err_deriv,
                this_dat_name)
