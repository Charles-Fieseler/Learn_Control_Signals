using Distributions
using StatsBase, Clustering
using MLBase # Just for cross validation

################################################################################
#####
##### Helper functions
#####
"""
Gets number of degrees of freedom, which is equal to the
number of nonzero terms + 1 for noise
"""
my_dof(m::sindyc_model) = length(get_nonzero_terms(m)) + 1


#####
##### Information theory
#####
"""
Akaike Information Criterion (AIC) for SINDy models
"""
my_aic(m::sindyc_model, dat, predictor; dist=Normal()) =
    -2loglikelihood(dist, (m(dat) .- predictor)') + 2my_dof(m)


"""
Corrected Akaike Information Criterion.
    Used for small sample sizes (Hurvich and Tsai 1989)
"""
function my_aicc(m::sindyc_model, dat, predictor; dist=Normal())
    k = my_dof(m)
    n = length(dat) # equal to numel(dat)
    correction = 2k*(k+1)/(n-k-1)
    return my_aic(m, dat, predictor, dist=dist) + correction
end


"""
Akaike Information Criterion (AIC), without being 'clever' and using an error
    distribution as in the above function
"""
simple_aic(m::sindyc_model, dat, predictor; dist=Normal()) =
    2log(my_rss(m, dat, predictor)) + 2my_dof(m)


################################################################################
#####
##### Cross validation
#####

"""
Computes the residual sum of squares error for a SINDYc model
"""
my_rss(m::sindyc_model, dat, predictor) = sum(m(dat, 0) .- predictor).^2;

"""
Computes the k-folds cross validation of a SINDYc model
    From: https://mlbasejl.readthedocs.io/en/latest/crossval.html
"""
function sindy_cross_validate(m::sindyc_model, dat, predictor; dist=Normal())
    n = size(dat, 2);
    scores = cross_validate(
        train_inds -> m(dat[:, train_inds], 0),        # training function
        # TODO: the rss shouldn't need the data itself
        (c, test_inds) -> my_rss(m, dat[:, test_inds], predictor[:, test_inds]),  # evaluation function
        n,              # total number of samples
        Kfold(n, 3))    # cross validation plan: 3-fold
    return mean(scores)
end

# sindy_cross_validate(m::sindyc_model, dat, predictor; dist=Normal()) =
#     sindy_cross_validate(m, dat, predictor)

################################################################################
#####
##### Ensemble testing functions
#####
"""
sindyc_ensemble(X, X_grad, library, val_list;
                U=nothing,
                selection_criterion=my_aic,
                selection_dist=Normal(),
                sparsification_mode="quantile",
                use_clustering_minimization=false)

Loop over several values of sparsity and choose the best model via
    MINIMIZING the 'selection_criterion' function
        selection_criterion() should have signature:
        selection_criterion(model, data, predictor)

    Also returns all the sparsity values and all the models
"""
function sindyc_ensemble(X, X_grad, library, val_list;
                U=nothing, ts=nothing,
                selection_criterion=my_aic,
                selection_dist=Normal(),
                sparsification_mode="quantile",
                use_clustering_minimization=false)
    if ts == nothing
        ts = 1:size(X,2)
    end
    n = length(val_list)
    all_models = Vector(undef, n)
    all_criteria = zeros(n)
    # Convenience function
    make_model(x) = if sparsification_mode == "quantile"
        sindyc(X, X_grad, U; library=library,
                                use_lasso=true,
                                quantile_threshold=x)
    elseif sparsification_mode == "num_terms"
        sindyc(X, X_grad, U, ts; library=library,
                                use_lasso=true,
                                quantile_threshold=nothing,
                                num_terms=x)
    else
        error("Unknown sparsification mode")
    end

    for (i, val) in enumerate(val_list)
        m = make_model(val) # Defined above with if statement
        all_criteria[i] = selection_criterion(m,
                X, X_grad, dist=selection_dist)
        all_models[i] = m
    end

    if !use_clustering_minimization
        best_criterion, best_index = findmin(all_criteria)
    else
        # Use AIC, but get the sparsest of the similar models
        #   Very helpful when the noise is highly Non-Gaussian
        tmp = reshape(all_criteria, length(all_criteria),1)
        c = kmeans(tmp', 6) # TODO: learn the number of clusters
        ind = findmin(all_criteria)[2]
        best_model_clust = c.assignments[ind]
        clust_ind_in_all_models = findall(c.assignments.==best_model_clust)
        nnz = sum.(val_list)[clust_ind_in_all_models]
        best_criterion, model_ind_in_clust = findmin(nnz)
        best_index = clust_ind_in_all_models[model_ind_in_clust]
    end

    return (best_model=all_models[best_index],
            best_criterion=best_criterion,
            all_criteria=all_criteria,
            all_models=all_models,
            best_index=best_index)
end

"""
An options struct to make it easier to call sindyc_ensemble
"""
function get_sindyc_ensemble_parameters()
    return Dict(:U=>nothing,
        :selection_criterion=>my_aicc,
        :selection_dist=>Normal(),
        :sparsification_mode=>"num_terms", # NOTE: different from raw object above
        :use_clustering_minimization=>false)
end


#
export sindyc_ensemble, get_sindyc_ensemble_parameters,
        my_aic, my_aicc, my_dof
