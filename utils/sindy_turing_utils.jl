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
                                noise_prior=Normal(5, 5.0))
    n_in, n_aug = size(model.A)
    total_terms = n_in * n_aug

    prior_coef = sindy_mat2coef(model.A)

    @model turing_model(y, dat) = begin
        # Set the priors using the passed model parameters
        all_coef ~ MvNormal(prior_coef, 1 .* ones(total_terms))
        A = sindy_coef2mat(model, all_coef)
        noise ~ Truncated(noise_prior, 0, 100)
        # Predict the gradient at each time step (grid collocation)
        for i in 1:size(y,2)
            preds = sindy_predict(model, all_coef, dat[:,i])
            y[:, i] ~ MvNormal(preds, noise.*ones(n_in))
        end
    end;

    return turing_model
end

#####
##### Helper functions for vector and matrix conversion
#####
function sindy_coef2mat(model::sindyc_model, coef::AbstractVector)
    return reshape(coef, size(model.A))
end

function sindy_mat2coef(mat::AbstractMatrix)
    n, m = size(mat)
    return reshape(mat, (n*m))
end

#####
##### Helper functions for readable prediction
#####
function sindy_predict(model::sindyc_model, coef::AbstractMatrix,
                        dat::AbstractVecOrMat)
    dat_aug = augment_data(model, dat)
    predictions = coef * dat_aug
    return predictions
end

# Vector input version
sindy_predict(model::sindyc_model, coef::AbstractVector, dat::AbstractVecOrMat) =
    sindy_predict(model, sindy_coef2mat(model, coef), dat)

#####
##### Creating model instances from chains
#####
function sindy_from_chain(m0::sindyc_model, chain::MCMCChains.AbstractChains)
    p_sample = sample(chain, 2) # sample size of 1 breaks
    p = Array(p_sample)[1,1:end-1] # Leave out noise
    # Assume these parameters are the A matrix
    A = sindy_coef2mat(m0, p)
    new_model = sindyc_model(A,
            m0.B, m0.U, m0.U_func, m0.library, m0.variable_names)
    return new_model
end

export convert_sindy_to_turing, sindy_from_chain
