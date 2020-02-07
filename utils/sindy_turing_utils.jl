using Turing, Distributions

# include("../src/sindyc.jl")
# include("sindy_utils.jl")

#####
##### Helper functions for defining a Turing model from a SINDy struct
#####
"""
Takes in a SINDy model and returns a Turing-compatible model that defines
    priors on each parameter with the proper variable names

Note: helpful NN analogue: https://turing.ml/dev/tutorials/3-bayesnn/
"""
function convert_sindy_to_turing(model::sindycModel;
                                dat_noise_prior=Normal(5, 5.0),
                                coef_noise_std=1.0)
    n_in, n_aug = size(model.A)
    total_terms = n_in * n_aug

    prior_coef = sindy_mat2vec(model.A)

    @model turing_model(y, dat) = begin
        # Set the priors using the passed model parameters
        all_coef ~ MvNormal(prior_coef, coef_noise_std .* ones(total_terms))
        A = sindy_vec2mat(model, all_coef)
        noise ~ Truncated(dat_noise_prior, 0, 100)
        # Predict the gradient at each time step (grid collocation)
        for i in 1:size(y,2)
            preds = sindy_predict(model, all_coef, dat[:,i])
            y[:, i] ~ MvNormal(preds, noise.*ones(n_in))
        end
    end;

    return turing_model
end

"""
Takes in a SINDy model and returns a Turing-compatible model that defines
    priors on each parameter with the proper variable names
    Enforces the 0s of the original model

    convert_sindy_to_turing_enforce_zeros(model::sindycModel;
                                    dat_noise_prior=Normal(5, 5.0),
                                    coef_noise_std=1.0)
"""
function convert_sindy_to_turing_enforce_zeros(model::sindycModel;
                                dat_noise_prior=Normal(5, 5.0),
                                coef_noise_std=1.0)
    n_in, n_aug = size(model.A)

    @model turing_model(y, dat) = begin
        term_ind = get_nonzero_terms(model)
        term_val = model.A[term_ind]
        all_coef ~ MvNormal(term_val, coef_noise_std)
        noise ~ Truncated(dat_noise_prior, 0, 100)
        # Predict the gradient at each time step (grid collocation)
        for i in 1:size(y,2)
            preds = sindy_predict(model, all_coef, dat[:,i],
                        nonzero_terms=term_ind)
            y[:, i] ~ MvNormal(preds, noise.*ones(n_in))
            # y[:, i] ~ MvNormal(preds, [noise, noise, noise])
        end
    end;

    return turing_model
end

#####
##### Helper functions for vector and matrix conversion
#####
function sindy_vec2mat(model::sindycModel, coef::AbstractVector)
    return reshape(coef, size(model.A))
end

function sindy_mat2vec(mat::AbstractMatrix)
    return reshape(mat, length(mat))
end

#####
##### Helper functions for readable prediction
#####
"""Skips full matrix multiplication; only does the nonzero coefficients"""
function sindy_predict(model::sindycModel, coef, dat; nonzero_terms=nothing)
    if nonzero_terms==nothing
        nonzero_terms=get_nonzero_terms(model, linear_indices=false)
    end
    dat_aug = augment_data(model, dat)
    original_ind = CartesianIndices(model.A)
    predictions = zeros(eltype(coef), size(dat))
    for (coef_ind,mat_ind) in enumerate(nonzero_terms)
        var, dat_ind = Tuple(mat_ind)
        predictions[var] += coef[coef_ind] * dat_aug[dat_ind]
    end
    return predictions
end

#####
##### Creating model instances from chains
#####
"""
function sindy_from_chain(model_template::sindycModel,
                          chain::MCMCChains.AbstractChains;
                          enforced_zeros=true)

Creates a sindycModel object (nonlinear dynamical system) from
a chain using the template of model terms in 'model_template';
    If enforced_zeros=true, then the zeros of the 'model_template'
    are enforced; the nonzero terms must match the number of
    elements in the chain minus one (the last one is assumed to be noise)
"""
function sindy_from_chain(model_template::sindycModel,
                          chain::MCMCChains.AbstractChains;
                          enforced_zeros=true)
    if enforced_zeros
        nonzero_terms = get_nonzero_terms(model_template)
        # @warn("model_template should have the same
        #         nonzeros structure as the desired model!")
    end

    ts = 1:length(model_template.U) # TODO: multidimensional U

    n_total = length(model_template.A)
    coef_sample = sample(chain, 2) # sample size of 1 gives error
    coef_vec = Array(coef_sample)[1,1:end-1] # Leave out noise
    subs2ind = CartesianIndices(model_template.A)
    if length(coef_vec) < n_total
        # Assume they are a subset of the A matrix, and
        # the proper index is in their name
        #   Add zeros to the other parameters
        var_names = coef_sample.name_map[:parameters][1:end-1]
        coef_tmp = zeros(n_total)
        for i in 1:n_total
            if enforced_zeros
                match_ind = findall(x->subs2ind[i]==x, nonzero_terms)
                found_match = (length(match_ind)==1)
            else
                # If no explicit mapping
                match_ind = occursin.("[$i]", var_names)
                found_match = any(match_ind)
            end
            if found_match
                coef_tmp[i] = coef_vec[match_ind][1]
            end
        end
        coef_vec = coef_tmp
    end
    # Assume these parameters are the A matrix
    A = sindy_vec2mat(model_template, coef_vec)
    new_model = sindycModel(ts, A,
            model_template.B, model_template.U,
            model_template.U_func, model_template.library,
            model_template.variable_names)
    return new_model
end


"""
Generate test realizations of the gradient from the posterior of the system
    parameters. No integration is performed
    Additional function to work with SINDy models
"""
function sample_sindy_posterior_grad(chain, dat, sample_ind, sindy_template;
                                num_samples=100, t = [0], enforced_zeros=true)
    sindy_samples = [sindy_from_chain(sindy_template, chain, enforced_zeros=enforced_zeros)
                        for i in 1:num_samples]
    all_vals = zeros(num_samples, length(sample_ind), size(dat,1))
    for i in 1:num_samples
        this_model = sindy_samples[i]
        for (i_save, i_dat) in enumerate(sample_ind)
            all_vals[i, i_save, :] = this_model(dat[:,i_dat], t)
        end
    end
    all_noise = Array(sample(chain[:noise], num_samples))

    return (sample_trajectories=all_vals, sample_noise=all_noise)
end



##
export convert_sindy_to_turing, convert_sindy_to_turing_enforce_zeros,
        sindy_from_chain, sample_sindy_posterior_grad
