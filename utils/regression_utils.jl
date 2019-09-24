
"""
Row-by-row sparsification using a quantile threshold
"""
function sparsify_signal!(U_guess; quantile_threshold=0.95)
    sparsify(x, eps) = abs(x) > eps ? x : 0;
    for i in 1:size(U_guess, 1)
        tmp = abs.(U_guess[i, :])
        epsilon = quantile(tmp[tmp.>0.0], quantile_threshold)
        for i2 in 1:size(U_guess,2)
            U_guess[i,i2] = sparsify(U_guess[i,i2], epsilon)
        end
    end
end

"""
Row-by-row sparsification using a quantile threshold
"""
function sparsify_signal(U_guess; quantile_threshold=0.95)
    for i in 1:size(U_guess, 1)
        tmp = abs.(U_guess[i, :])
        epsilon = quantile(tmp[tmp.>0.0], quantile_threshold)
        ind = abs.(U_guess[i, :]) .< epsilon
        U_guess[i, ind] .= 0.0
    end
    return U_guess
end

"""
Row-by-row sparsification using a hard threshold
"""
function threshold_signal!(U_guess; hard_threshold=1e-5)
    sparsify(x, eps) = abs(x) > eps ? x : 0;
    for i in 1:size(U_guess, 1)
        tmp = abs.(U_guess[i, :])
        for i2 in 1:size(U_guess,2)
            U_guess[i,i2] = sparsify(U_guess[i,i2], hard_threshold)
        end
    end
end

"""
Simple implementation of sequential least squares thresholding for a linear
    regression model
"""
function sparse_regression(X::Matrix, y::Matrix;
                           num_iter=5, quantile_threshold=0.1,
                           hard_threshold=nothing)
    if size(y,1) > 1
        A = zeros(size(y,1), size(X,1))
        # For the 2d predictor case, just loop over rows
        for i in 1:size(y,1)
            global A[i, :] = sparse_regression(X, y[i:i,:],
                            num_iter=num_iter,
                            quantile_threshold=quantile_threshold,
                            hard_threshold=hard_threshold)
        end
        return A
    end
    # Initialize with L2
    A = y/X
    sparsify_signal!(A, quantile_threshold=quantile_threshold)
    # Redo, removing the rows in X that were zeroed out
    for i in 1:num_iter
        which_rows = vec(abs.(A) .> 0)
        A_subset = y/(X[which_rows, :])
        global A[which_rows] = A_subset
        global A[.!which_rows] .= 0.0
        if hard_threshold == nothing
            sparsify_signal!(A, quantile_threshold=quantile_threshold)
        else
            threshold_signal!(A, hard_threshold=hard_threshold)
        end
    end

    return A
end



export sparse_regression, sparsify_signal, threshold_signal!
