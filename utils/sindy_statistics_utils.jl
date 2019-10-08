using Distributions

#####
##### Helper functions
#####
"""
Gets number of degrees of freedom, which is equal to the
number of nonzero terms + 1 for noise
"""
function dof(m) = length(get_nonzero_terms(m)) + 1


#####
##### Information theory
#####
"""
Akaike Information Criterion (AIC) for SINDy models
"""
function aic(m::sindyc_model, dat) =
    -2loglikelihood(MvNormal(), (m(dat) .- dat)') + 2dof(m)


"""
Corrected Akaike Information Criterion.
    Used for small sample sizes (Hurvich and Tsai 1989)
"""
function aicc(m::sindyc_model, dat)
    k = dof(m)
    n = length(dat) # equal to numel(dat)
    correction = 2k*(k+1)/(n-k-1)
    return aic(m, dat) + correction
end


#####
##### Ensemble testing functions
#####
"""
Loop over several values of sparsity and choose the best model via
    MINIMIZING the 'selection_criterion' function
        selection_criterion() should have signature:
        selection_criterion(model, data)

    Also returns all the sparsity values and all the models
"""
function sindyc_ensemble(X, X_grad, library, quantile_list;
                selection_criterion=aic)
    n = length(quantile_list)
    all_models = Vector(undef, n)
    all_criteria = zeros(n)
    # Convinience function
    make_model(x) = sindyc(X, X_grad; library=library,
                            use_lasso=true,
                            quantile_threshold=x)

    for (i, val) in enumerate(quantile_list)
        m = make_model(val)
        all_criteria[i] = selection_criterion(m, X)
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
        aic, aicc, dof
