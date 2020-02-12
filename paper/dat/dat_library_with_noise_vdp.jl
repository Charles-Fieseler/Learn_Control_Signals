using PkgSRA
using Plots, Random, Distributions, Interpolations
using StatsPlots, StatsBase
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");

################################################################################
#####
##### FitzHugh Nagumo example: get data
#####
include(EXAMPLE_FOLDERNAME*"example_vanDerPol.jl")

# Define the multivariate forcing function
function local_make_data(num_ctr)
    num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0.2;
    amplitude = 2.0
    my_U_func_time2(t) = U_func_time(t, u0,
                            U_widths, U_starts,
                            F_dim=1,
                            amplitude=amplitude)
    # Get data
    sol = solve_vdp_system(U_func_time=my_U_func_time2)
    dat = Array(sol)
    numerical_grad = numerical_derivative(dat, ts)

    # Also get truth
    true_grad = core_dyn_true(dat)
    U_true = zeros(size(dat))
    for (i, t) in enumerate(ts)
        U_true[:,i] = my_U_func_time2(t)
    end

    ## Also get baseline true/ideal cases
    # Uncontrolled
    dat_raw = Array(solve_vdp_system())
    numerical_grad_raw = numerical_derivative(dat_raw, ts)
    true_grad_raw = zeros(size(dat))
    #
    # # Intialize truth object
    this_truth = sra_truth_object(dat_raw, true_grad, U_true, core_dyn_true)

    # End
    return this_truth, dat, numerical_grad
    # return dat, numerical_grad
end

#####
##### Loop over noise and number of spikes
#####
# Real values
# control_signal_vals = 0:3:12
# num_models = 40
# noise_vals = 0.0:0.05:0.5
# Shorter, test values
control_signal_vals = [0,5]
num_models = 3
noise_vals = 0.0:0.05:0.1

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
    variable_names = ["v", "ω"]
    optimizer = slstNumber((1, 1)); # Number of terms-based sparse regression
    library = Dict("cross_terms"=>[2, 3], "constant"=>nothing)
    model_template = sindyModel(convert_string2function(library),
        variable_names, optimizer)
    this_model.parameters = get_sra_defaults(false)
    prams = this_model.parameters
    prams.model_template = model_template
    prams.sindy_terms_list = Iterators.product(1:4, 1:4)
    prams.initial_subsampling = true
end

for (i_noise,σ) in enumerate(noise_vals)
    for (i_ctr, num_ctr) in enumerate(control_signal_vals)
    this_truth, dat, numerical_grad = local_make_data(num_ctr)
        for i_model in 1:num_models
            # Get data
            noisy_grad = numerical_grad .+ σ.*randn(size(dat))
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
                    this_model.sindy_model, dat, numerical_grad) ./ length(dat)

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
                    this_model.sindy_model, dat[:,ind], numerical_grad[:,ind]) ./
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

#####
##### Save data
#####
this_dat_name = "dat_library_with_noise_vdp_"


save_for_plotting(noise_vals, control_signal_vals, coef_norm,
                all_naive_err, all_err,
                all_err_deriv_subsample, all_naive_err_deriv,
                this_dat_name)
