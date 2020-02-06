
###
### Structs that define the solver settings
###
abstract type sparse_solver end

# Sequential Least Squares methods
struct slst_hard <: sparse_solver
    num_iter::Number
    hard_threshold::Number
end
slst_hard(hard_threshold) = slst_hard(1, hard_threshold)

struct slst_quantile <: sparse_solver
    num_iter::Number
    quantile_threshold::Number
end
slst_hard(quantile_threshold) = slst_hard(1, quantile_threshold)

struct slst_number <: sparse_solver
    num_iter::Number
    num_terms::Vector
end
slst_number(num_terms) = slst_number(1, num_terms)

struct dense_solver <: sparse_solver end

###
### Actually solving
###
"""
Uses sparse_solver objects and implements Sequential Least Squares Thresholding
    in various ways

This is the outer user-facing function that performs checks, namely whether
    recursion over rows is needed
"""
function sparse_regression(alg::sparse_solver, X::Matrix, y::Matrix)
                           # num_iter=2,
                           # quantile_threshold=0.1,
                           # hard_threshold=nothing,
                           # num_terms=nothing)
    # Initial check: should we do recursive?
    if size(y,1) > 1
        A = zeros(size(y,1), size(X,1))
        # For the 2d predictor case, just loop over rows
        for i in 1:size(y,1)
            # if num_terms !== nothing
            #     # Constrain the number of terms row by row
            #     if length(num_terms)>1
            #         n = num_terms[i]
            #     else
            #         n = num_terms
            #     end
            # else
            #     n = nothing
            # end
            # Call inner function
            this_y = y[i:i,:]
            A[i, :] = _sparse_regression(alg, X, this_y, i)#,
                            # num_iter=num_iter,
                            # quantile_threshold=quantile_threshold,
                            # hard_threshold=hard_threshold,
                            # num_terms=n)
        end
        return A
    end

    A =  _sparse_regression(alg, X, y, nothing)
    return A
end



###
### Private functions
###
"""
Fake "sparse" solver; just L2 solution
"""
function _sparse_regression(alg::dense_solver, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    return X/y
end

"""
Implements a hard threshold

see also: threshold_signal!
"""
function _sparse_regression(alg::slst_hard, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = X/y
    end
    # Redo, removing the rows in X that were zeroed out
    for i in 1:alg.num_iter
        threshold_signal!(A, hard_threshold=alg.hard_threshold)
        which_rows = vec(abs.(A) .> 0)
        A_subset = y/(X[which_rows, :])
        A[which_rows] = A_subset
        A[.!which_rows] .= 0.0 # Retain the previous zeros
    end

    return A
end


"""
Implements a quantile threshold

see also: sparsify_signal!
"""
function _sparse_regression(alg::slst_quantile, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = X/y
    end
    # Redo, removing the rows in X that were zeroed out
    for i in 1:alg.num_iter
        sparsify_signal!(A, quantile_threshold=alg.quantile_threshold)
        which_rows = vec(abs.(A) .> 0)
        A_subset = y/(X[which_rows, :])
        A[which_rows] = A_subset
        A[.!which_rows] .= 0.0 # Retain the previous zeros
    end

    return A
end

"""
Implements a hard cap on the number of terms allowed

see also: keep_n_terms
"""
function _sparse_regression(alg::slst_number, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = X/y
    end
    # Redo, removing the rows in X that were zeroed out
    for i in 1:alg.num_iter
        A = keep_n_terms(A, alg.num_terms[which_data_row])
        which_rows = vec(abs.(A) .> 0)
        A_subset = y/(X[which_rows, :])
        A[which_rows] = A_subset
        A[.!which_rows] .= 0.0 # Retain the previous zeros
    end

    return A
end


export slst_hard, slst_quantile, slst_number, dense_solver, sparse_solver,
    sparse_regression
