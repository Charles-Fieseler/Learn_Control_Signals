# Core data structure for the state of an SRA analysis
#   See also: sra_model_plotting.jl for plotting methods
#   See also: sra_model_function.jl for the analysis steps

#####
##### Helper structure, for the analysis parameters
#####
struct sra_parameters
    debug_mode::Bool
    debug_folder::String
    # Initial, naive model
    initial_subsampling::Bool
    # Calculating residuals of the previous model
    use_turing::Bool
    noise_factor::Number
    # Parameters shared between all iterations of SINDy model
    variable_names
    sindy_library::Dict
    sindy_terms_list::Vector
    sindyc_ensemble_parameters
    # Subsampling to get the next iteration
    num_pts::Number
    start_ind::Number
end

# Initializer with defaults
function get_sra_defaults()
    debug_mode = true;
    debug_folder = nothing;
    # Initial, naive model
    initial_subsampling = false;
    # Calculating residuals of the previous model
    use_turing = false;
    noise_factor = 1.0;
    # Parameters shared between all iterations of SINDy model
    variable_names = ["x", "y", "z"]
    sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
    sindy_terms_list = calc_permutations(6,3);
    sindyc_ensemble_parameters = get_sindyc_ensemble_parameters();
    # Subsampling to get the next iteration
    num_pts = 1000;
    start_ind = 1;

    return sra_parameters(debug_mode, debug_file,
        initial_subsampling,
        use_turing,
        noise_factor,
        sindy_library,
        sindy_terms_list,
        sindy_minimization_options,
        num_pts,
        start_ind)
end


#####
##### Helper structure, for the true values
#####
struct sra_truth_object
    true_grad::Array
    U_true
    core_dyn_true
end


#####
##### Main structure, for the current state
#####
struct sra_stateful_object
    is_saved::Bool
    parameters::sra_parameters
    # Initial data and gradients
    ts
    tspan
    dat::Array
    u0::Vector # Note: initial conditions may not be same as training data
    numerical_grad::Array
    # Analysis history
    all_sindy_models
    all_ensemble_sindy_criteria
    all_best_sindy_criteria
    all_sindy_dat
    all_residuals
    all_subsample_ind
    all_noise_guesses
    # Current analysis state
    sindy_model
    ensemble_sindy_criteria
    best_sindy_criteria
    sindy_dat
    residual
    subsample_ind
    noise_guess
end

# Initializer with defaults
function sra_stateful_object(ts, tspan, dat, u0, numerical_grad)
    return sra_stateful_object(
        true,
        get_sra_defaults(),
        ts, tspan, dat, u0, numerical_grad,
        [], [], [], [], [], [], [],
        nothing, nothing, nothing, nothing, nothing, nothing, nothing
        )
end

export sra_stateful_object, sra_truth_object,
    sra_parameters, get_sra_defaults
