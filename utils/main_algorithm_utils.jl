using PkgSRA


"""
Steps 1 and 3: fit a distribution of models

calc_distribution_of_models(dat,
                             numerical_grad,
                             sindy_library;
                             U=nothing,
                             selection_dist=Normal(0,10),
                             use_clustering_minimization=false,
                             chain_opt=(iterations=200,
                                        num_training_pts=100,
                                        start_ind=101),
                                        val_list = Iterators.product(1:3,1:3))

Returns both the Turing chain and the sindyc_model used as the
        priors for the coefficients.
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
                                                val_list = Iterators.product(1:3,1:3)) # DEBUG
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

calc_distribution_of_residuals(dat, numerical_grad, chain,
                                sample_ind,
                                sindy_template)

Returns:
        residual
        sample_gradients
        sample_noise
        dat_grad
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

function subsample_using_residual(residual, noise;
                                  noise_factor=1.0,
                                  min_length=1,
                                  shorten_length=1)
return accepted_ind
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



"""
Optional step 0: Randomly subsample the initial data (for the naive model)

calc_best_random_subsample(dat2, numerical_grad2, sindy_library;
                                    num_pts=400,
                                    num_subsamples=10,
                                    val_list = Iterators.product(1:3,1:3))
return (best_initial_subsample=best_initial_subsample,
        best_model=all_final_models2[tmp_ind],
        all_initial_subsamples=initial_subsamples,
        all_errL2=all_errL2)
"""
function calc_best_random_subsample(dat2, numerical_grad2, sindy_library;
                        num_pts=400,
                        num_subsamples=10,
                        val_list = Iterators.product(1:3,1:3),
                        sindyc_ensemble_params=get_sindyc_ensemble_parameters())
    # Initially, randomly subsample the data
    initial_subsamples = []
    for i in 1:num_subsamples
        push!(initial_subsamples, randperm(size(dat2,2))[1:num_pts])
    end

    # Just do SINDY here, not Turing yet
    sz = size(initial_subsamples)
    all_final_models2 = Vector{sindyc_model}(undef,sz)
    all_errL2 = zeros(sz)
    for (i, inds) in enumerate(initial_subsamples)
        (all_final_models2[i], all_errL2[i], _, _, _) =
             sindyc_ensemble(dat2[:,inds],
                             numerical_grad2[:, inds],
                             sindy_library, val_list;
                             sindyc_ensemble_params...)
        if sindyc_ensemble_params[:selection_criterion] == my_aicc
            # Calculate real L2 error
            this_dat = all_final_models2[i](dat2, 0)
            all_errL2[i] = sum(abs, numerical_grad2.-this_dat)
        else
            # Should be cross validation; already L2 error
        end
    end

    # Choose the best of the above models and subsamples
    min_err, tmp_ind = findmin(all_errL2)
    best_initial_subsample = initial_subsamples[tmp_ind]

    return (best_initial_subsample=best_initial_subsample,
            best_model=all_final_models2[tmp_ind],
            all_initial_subsamples=initial_subsamples,
            all_errL2=all_errL2)
end


#####
#####
#####
export calc_distribution_of_models,
        calc_distribution_of_residuals,
        subsample_using_residual,
        calc_best_random_subsample
