using PkgSRA


"""
Steps 1 and 3: fit a distribution of models
"""
function calc_distribution_of_models(dat,
                                     numerical_grad,
                                     sindy_library;
                                     U=nothing,
                                     selection_dist=Normal(0,10),
                                     use_clustering_minimization=false,
                                     chain_opt=(iterations=200,
                                                num_training_pts=100,
                                                start_ind=101),
                                     val_list=calc_permutations(5,2))
   # Use AIC to find the best model
   (best_sindy_model,best_criterion,all_criteria,all_models, best_index) =
       sindyc_ensemble(dat, numerical_grad, sindy_library, val_list,
                       U=U,
                       selection_criterion=my_aicc,
                       sparsification_mode="num_terms",
                       selection_dist=selection_dist,
                       use_clustering_minimization=use_clustering_minimization)

   # Use Turing to fit a distribution of models
   turing_model = convert_sindy_to_turing_enforce_zeros(best_sindy_model;
                                   dat_noise_prior=Normal(0.0, 20.0),
                                   coef_noise_std=1.0)
   chain = generate_chain(dat, numerical_grad,
                          turing_model;
                          chain_opt...)[1]

   return (chain=chain, best_sindy_model=best_sindy_model)
end


"""
Step 2: Get the distribution of residuals
"""
function calc_distribution_of_residuals(dat, numerical_grad, chain,
                                        sample_ind,
                                        sindy_template)
    sample_gradients3d, sample_noise2d = sample_sindy_posterior_grad(chain,
                dat, sample_ind, sindy_template)

    # Calculate the residuals
    sample_gradients = transpose(drop_all_1dims(mean(sample_gradients3d, dims=1)))
        # sample_gradients_std = transpose(drop_all_1dims(std(sample_gradients3d, dims=1)))
        sample_noise = mean(sample_noise2d)
    dat_grad = numerical_grad[:, sample_ind]

    residual = dat_grad .- sample_gradients

    return (residual=residual,
            sample_gradients=sample_gradients,
            sample_noise=sample_noise,
            dat_grad=dat_grad)
end



"""
Step 3: Subsample the data using the residuals
"""
function subsample_using_residual(residual, noise;
                                  noise_factor=1.0,
                                  min_length=1,
                                  shorten_length=1)
    partial_accepted_ind = abs.(residual) .< (noise_factor*noise)
    partial_accepted_ind = vec(prod(Int.(partial_accepted_ind), dims=1))
    _, _, all_starts, all_ends = calc_contiguous_blocks(
            partial_accepted_ind,
            minimum_length=min_length + 2*shorten_length)

    # Get the "good" indices
    accepted_ind = []
    for (s, e) in zip(all_starts, all_ends)
        accepted_ind = vcat(accepted_ind,
                (s+shorten_length):(e-shorten_length))
    end
    return accepted_ind
end



export calc_distribution_of_models,
        calc_distribution_of_residuals,
        subsample_using_residual
