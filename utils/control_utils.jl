"""
Takes a time series and the noise envelope, and returns the values of the time
series that are "significant" i.e. outside the noise envelope
"""
function process_residual(mean, std)
    low = mean .- std
    high = mean .+ std
    real_ind = .!((low .< 0.0) .& (high .> 0.0))

    ctr = zeros(size(mean))
    ctr[real_ind] = mean[real_ind]

    return ctr
end

export process_residual
