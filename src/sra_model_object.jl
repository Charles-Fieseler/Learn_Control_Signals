# Core data structure for the state of an SRA analysis
#   See also: sra_model_plotting.jl for plotting methods

#####
##### Helper structure, for the analysis parameters
#####
struct sra_parameters
    debug_mode::Bool
    debug_file::String
    # Initial, naive model
    initial_subsampling::Bool
    # Calculating residuals of the previous model
    use_turing::Bool
    noise_factor::Number
    # Parameters shared between all iterations of SINDy model
    sindy_library::Dict
    sindy_terms_list::Vector
    sindy_minimization_options
    # Subsampling to get the next iteration
    num_pts::Number
    start_ind::Number
end

# Initializer with defaults
function get_sra_defaults()
    debug_mode = true;
    debug_file = nothing;
    # Initial, naive model
    initial_subsampling = false;
    # Calculating residuals of the previous model
    use_turing = false;
    noise_factor = 1.0;
    # Parameters shared between all iterations of SINDy model
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
struct sra_stateful_object
    true_grad::Array
    U_true
    core_dyn_true
end


#####
##### Main structure, for the current state
#####
struct sra_stateful_object
    # Initial data and gradients
    dat::Array
    numerical_grad::Array
    # Analysis history
    all_sindy_models
    all_ensemble_sindy_criteria
    all_best_sindy_criteria
    all_residuals
    all_accepted_ind
    # Current analysis state
    sindy_model
    ensemble_sindy_criteria
    best_sindy_criteria
    residual
    accepted_ind
end
