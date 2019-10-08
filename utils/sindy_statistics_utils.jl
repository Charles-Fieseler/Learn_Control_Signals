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
