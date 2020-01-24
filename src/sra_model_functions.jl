# Plotting the data structure for the state of an SRA analysis
#   See also: sra_model_object.jl for the saved states
#   See also: sra_model_plotting.jl for plotting methods
include("../utils/main_algorithm_utils.jl")
include("../paper/scripts/paper_settings.jl");
using BSON: @save

#####
##### Model fitting
#####
"""
Step 1a: Fit a naive sindy model to the data
    Note: only called first time around

Sensitive to parameters:
noise_factor
initial_subsampling
use_turing

See methods:
print_current_equations
print_true_equations

Expectation: this model will not be very good!
"""
function fit_first_model(m::sra_stateful_object, initial_noise)
    # TODO: move noise level to parameter object
    p = m.parameters
    # Change noise estimate based on previous guess
    ensemble_p = p.sindyc_ensemble_parameters
    # ensemble_p._replace(selection_dist=Normal(0.0, initial_noise))
    ensemble_p[:selection_dist] = Normal(0.0, initial_noise)
    (sindy_model,best_criterion,all_criteria,all_models) =
        sindyc_ensemble(
                m.dat,
                m.numerical_grad,
                p.sindy_library,
                p.sindy_terms_list;
                var_names=p.variable_names,
                ensemble_p...)
    m.sindy_model = sindy_model
    m.ensemble_sindy_criteria = all_criteria
    m.best_sindy_criteria = best_criterion
    m.i = 1 # Resets counter
end

"""
Step 1b: Fit a model based on subsampled data
    Note: Mostly same as above, for non-initial iterations

Sensitive to parameters:
TODO: noise_factor

See methods:
print_current_equations
print_true_equations
plot_subsampled_derivatives
plot_subsampled_simulation

Expectation: this model will not be very good!
"""
function fit_model(m::sra_stateful_object)
    if !m.is_saved
        save_model_variables(m)
    end
    p = m.parameters
    # New: change noise estimate based on previous guess
    ensemble_p = p.sindyc_ensemble_parameters
    ensemble_p[:selection_dist] = Normal(0.0, m.noise_guess)
    # Same
    (sindy_model,best_criterion,all_criteria,all_models) =
        sindyc_ensemble(
                m.dat[:, m.subsample_ind],
                m.numerical_grad[:, m.subsample_ind],
                p.sindy_library,
                p.sindy_terms_list;
                var_names=p.variable_names,
                ensemble_p...)
    # These overwrite the previous step, which has been saved
    m.sindy_model = sindy_model
    m.sindy_dat = nothing # Reset
    m.ensemble_sindy_criteria = all_criteria
    m.best_sindy_criteria = best_criterion
    m.is_saved = false
    m.i += 1
    return all_models # DEBUG
end


#####
##### Residual calculation
#####
"""
Step 2: Subsample based on previous model residuals

Sensitive to parameters:
noise_factor
use_turing

See methods:
plot_subsampled_points
"""
function calculate_subsampled_ind(m::sra_stateful_object)
    p = m.parameters
    if !m.parameters.use_turing
        sindy_grad = m.sindy_model(m.dat, 0)
        residual = m.numerical_grad .- sindy_grad
        # TODO: don't hardcode the index here
        noise_guess = p.noise_factor*abs(median(residual[1,:])) # Really should use Turing here
    else
        error("Turing not implemented in stateful version")
    end
    # TODO: don't hardcode length (and other parameters)
    # TODO: Should this really be a loop?
    accepted_ind = []
    tmp_noise_factor = 1.0
    i = 1
    while length(accepted_ind) < p.num_pts
        accepted_ind = subsample_using_residual(residual,
                    tmp_noise_factor*noise_guess, min_length=4);
        i += 1;
        i > 10 && break
        tmp_noise_factor *= 1.5;
    end

    if length(accepted_ind) < p.num_pts+p.start_ind-1
        error("DataError: Not enough accepted points; increase noise_factor or decrease num_pts")
    end
    m.subsample_ind = accepted_ind[p.start_ind:p.num_pts+p.start_ind-1]
    m.noise_guess = noise_guess
end


#####
##### Other steps
#####
"""
Save current step for continuation
"""
function save_model_variables(m::sra_stateful_object)
    pushfirst!(m.all_sindy_models, m.sindy_model)
    pushfirst!(m.all_ensemble_sindy_criteria, m.ensemble_sindy_criteria)
    pushfirst!(m.all_best_sindy_criteria, m.best_sindy_criteria)
    pushfirst!(m.all_sindy_dat, m.sindy_dat)
    pushfirst!(m.all_residuals, m.residual)
    pushfirst!(m.all_subsample_ind, m.subsample_ind)
    pushfirst!(m.all_noise_guesses, m.noise_guess)

    m.is_saved = true;
end


#####
##### Other steps
#####
"""
Side-Step: Integrate the current model

Note: not necessary for simulation, just for plotting

See methods:
plot_sindy_model
"""
function simulate_model(m::sra_stateful_object)
    p = m.parameters
    # Callback aborts if it blows up
    condition(u,t,integrator) = any(abs.(u).>1e4)
    cb = DiscreteCallback(condition, terminate!)
    prob = ODEProblem(m.sindy_model, m.u0, m.tspan, [0], callback=cb)
    m.sindy_dat = Array(solve(prob, Tsit5(), saveat=m.ts));
end


"""
Calculates the residual of the data and the current model
"""
function calc_residual(m::sra_stateful_object)
    sindy_grad = m.sindy_model(m.dat, 0)
    residual = m.numerical_grad .- sindy_grad
    return residual
end


"""
Saves the data in a standardized format in four different files

Note: requires the paper_settings.jl file to specify the DAT_FOLDERNAME variable

Usage:
save_for_plotting(m::sra_stateful_object,
                t::sra_truth_object,
                this_dat_name="test")
"""
function save_for_plotting(m::sra_stateful_object,
                            t::sra_truth_object,
                            this_dat_name="test")
    this_dat_name = DAT_FOLDERNAME*this_dat_name;

    dat_raw = t.dat_raw
    fname = this_dat_name*"uncontrolled.bson"
    @save fname dat_raw

    dat, U_true = m.dat, t.U_true
    fname = this_dat_name*"ode_vars.bson";
    @save fname dat U_true

    noise_guess, accepted_ind = m.noise_guess, m.subsample_ind
    residual = calc_residual(m)
    fname = this_dat_name*"naive_vars_bayes.bson";
    @save fname noise_guess residual accepted_ind #ctr_guess

    dat_ctr = simulate_model(m)
    ctr_guess2 = process_residual(residual, noise_guess) # TODO: should be Bayesian
    fname = this_dat_name*"ctr_vars_sindy.bson";
    @save fname dat_ctr ctr_guess2 #sindy_grad_ctr
end

##### Export
export fit_first_model, fit_model, calculate_subsampled_ind,
    save_model_variables, simulate_model, save_for_plotting,
    calc_residual
