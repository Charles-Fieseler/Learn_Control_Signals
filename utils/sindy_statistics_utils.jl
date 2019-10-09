using Distributions
using StatsBase

#####
##### Helper functions
#####
"""
Gets number of degrees of freedom, which is equal to the
number of nonzero terms + 1 for noise
"""
my_dof(m) = length(get_nonzero_terms(m)) + 1


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


#####
##### Ensemble testing functions
#####
"""
Loop over several values of sparsity and choose the best model via
    MINIMIZING the 'selection_criterion' function
        selection_criterion() should have signature:
        selection_criterion(model, data, predictor)

    Also returns all the sparsity values and all the models
"""
function sindyc_ensemble(X, X_grad, library, val_list;
                selection_criterion=my_aic,
                selection_dist=Normal(),
                sparsification_mode="quantile")
    n = length(val_list)
    all_models = Vector(undef, n)
    all_criteria = zeros(n)
    # Convinience function
    make_model(x) = if sparsification_mode == "quantile"
        sindyc(X, X_grad; library=library,
                                use_lasso=true,
                                quantile_threshold=x)
    elseif sparsification_mode == "num_terms"
        sindyc(X, X_grad; library=library,
                                use_lasso=true,
                                quantile_threshold=nothing,
                                num_terms=x)
    else
        error("Unknown sparsification mode")
    end

    for (i, val) in enumerate(val_list)
        m = make_model(val)
        all_criteria[i] = selection_criterion(m,
                X, X_grad, dist=selection_dist)
        all_models[i] = m
    end

    best_criterion, best_index = findmin(all_criteria)

    return (best_model=all_models[best_index],
            best_criterion=best_criterion,
            all_criteria=all_criteria,
            all_models=all_models)
end


#
export sindyc_ensemble,
        my_aic, my_aicc, my_dof
