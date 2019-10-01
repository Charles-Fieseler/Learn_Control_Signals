using Turing, Distributions

include("../src/sindyc.jl")
include("sindy_utils.jl")

#####
##### Helper functions for defining a Turing model from a SINDy struct
#####
"""
Takes in a SINDy model and returns a Turing-compatible model that defines
    priors on each parameter with the proper variable names

Note: helpful NN analogue: https://turing.ml/dev/tutorials/3-bayesnn/
"""
function convert_sindy_to_turing(model::sindyc_model;
                                dat_noise_prior=Normal(5, 5.0),
                                coef_noise_std=1.0)
    n_in, n_aug = size(model.A)
    total_terms = n_in * n_aug

    prior_coef = sindy_mat2vec(model.A)

    @model turing_model(y, dat) = begin
        # Set the priors using the passed model parameters
        all_coef ~ MvNormal(prior_coef, coef_noise_std .* ones(total_terms))
        A = sindy_vec2mat(model, all_coef)
        noise ~ Truncated(noise_prior, 0, 100)
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
"""
function convert_sindy_to_turing_enforce_zeros(model::sindyc_model;
                                dat_noise_prior=Normal(5, 5.0),
                                coef_noise_std=1.0)
    n_in, n_aug = size(model.A)
    total_terms = n_in * n_aug

    prior_coef = sindy_mat2vec(model.A)

    @model turing_model(y, dat) = begin
        # Set the priors using the passed model parameters
        all_coef = Array{Union{Real, Distribution}}(undef, total_terms)
        for (i, coef) in enumerate(prior_coef)
            if abs(prior_coef[i]) > 0.0
                all_coef[i] ~ Normal(prior_coef[i], coef_noise_std)
            else
                # all_coef[i] ~ Normal(0.0, 1e-4)
                all_coef[i] = 0.0
            end
        end
        A = sindy_vec2mat(model, all_coef)
        noise ~ Truncated(dat_noise_prior, 0, 100)
        # Predict the gradient at each time step (grid collocation)
        for i in 1:size(y,2)
            preds = sindy_predict(model, all_coef, dat[:,i])
            # TODO: does this loop actually work?
            for i_var in 1:n_in
                y[i_var, i] ~ Normal(preds[i_var], noise)
            end
            # y[:, i] ~ MvNormal(preds, noise.*ones(n_in))
        end
    end;

    return turing_model
end

#####
##### Helper functions for vector and matrix conversion
#####
function sindy_vec2mat(model::sindyc_model, coef::AbstractVector)
    return reshape(coef, size(model.A))
end

function sindy_mat2vec(mat::AbstractMatrix)
    n, m = size(mat)
    return reshape(mat, (n*m))
end

#####
##### Helper functions for readable prediction
#####
function sindy_predict(model::sindyc_model, coef::AbstractMatrix,
                        dat::AbstractVecOrMat)
    dat_aug = augment_data(model, dat)
    predictions = coef * convert.(eltype(coef),dat_aug)
    return predictions
end

# Vector input version
sindy_predict(model::sindyc_model, coef::AbstractVector, dat::AbstractVecOrMat) =
    sindy_predict(model, sindy_vec2mat(model, coef), dat)

#####
##### Creating model instances from chains
#####
function sindy_from_chain(m0::sindyc_model, chain::MCMCChains.AbstractChains)
    n_total = length(m0.A)

    coef_sample = sample(chain, 2) # sample size of 1 gives error
    coef_vec = Array(coef_sample)[1,1:end-1] # Leave out noise
    if length(coef_vec) < n_total
        # Assume they are a subset of the A matrix
        #   Add zeros to the other parameters
        var_names = coef_sample.name_map[:parameters][1:end-1]
        coef_tmp = zeros(n_total)
        for i in 1:n_total
            match_ind = occursin.("[$i]", var_names)
            if any(match_ind)
                coef_tmp[i] = coef_vec[match_ind][1]
            end
        end
        coef_vec = coef_tmp
    end
    # Assume these parameters are the A matrix
    A = sindy_vec2mat(m0, coef_vec)
    new_model = sindyc_model(A,
            m0.B, m0.U, m0.U_func, m0.library, m0.variable_names)
    return new_model
end

export convert_sindy_to_turing, sindy_from_chain
