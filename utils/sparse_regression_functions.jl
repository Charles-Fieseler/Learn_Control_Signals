
###
### Structs that define the solver settings
###
abstract type SparseSolver end

# Sequential Least Squares methods
struct slstHard <: SparseSolver
    num_iter::Number
    hard_threshold::Number
    iterate_through_rows::Bool
end
slstHard(num_iter, hard_threshold) = slstHard(num_iter, hard_threshold, true)
slstHard(hard_threshold) = slstHard(1, hard_threshold)

struct slstQuantile <: SparseSolver
    num_iter::Number
    quantile_threshold::Number
    iterate_through_rows::Bool
end
slstQuantile(num_iter, quantile_threshold) =
    slstQuantile(num_iter, quantile_threshold, true)
slstQuantile(quantile_threshold) = slstQuantile(1, quantile_threshold)

struct slstNumber <: SparseSolver
    num_iter::Number
    num_terms::Tuple
    iterate_through_rows::Bool
end
slstNumber(num_iter, num_terms) = slstNumber(num_iter, num_terms, true)
slstNumber(num_terms) = slstNumber(1, num_terms)

struct denseSolver <: SparseSolver end

copy_optimizer(s::SparseSolver, options...) = typeof(s)(options...)

###
### Actually solving
###
"""
Uses SparseSolver objects and implements Sequential Least Squares Thresholding
    in various ways

This is the outer user-facing function that performs checks, namely whether
    recursion over rows is needed
"""
function sparse_regression(alg::SparseSolver, X::Matrix, y::Matrix)
                           # num_iter=2,
                           # quantile_threshold=0.1,
                           # hard_threshold=nothing,
                           # num_terms=nothing)
    # Initial check: should we do recursive?
    if size(y,1) > 1 && alg.iterate_through_rows
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
            # println(size(this_y))
            A[i, :] = private_sparse_regression(alg, X, this_y, i)#,
                            # num_iter=num_iter,
                            # quantile_threshold=quantile_threshold,
                            # hard_threshold=hard_threshold,
                            # num_terms=n)
            # @show A
        end
    else
        A =  private_sparse_regression(alg, X, y, 0, nothing)
    end
    return A
end



###
### Private functions
###
"""
Fake "sparse" solver; just L2 solution
"""
function private_sparse_regression(alg::denseSolver, X::Matrix, y::Matrix,
                            A=nothing)
    return y/X
end

"""
Implements a hard threshold

see also: threshold_signal!
"""
function private_sparse_regression(alg::slstHard, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = y/X
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
function private_sparse_regression(alg::slstQuantile, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = y/X
    end
    # println(A)
    # Redo, removing the rows in X that were zeroed out
    for i in 1:alg.num_iter
        sparsify_signal!(A, quantile_threshold=alg.quantile_threshold)
        which_rows = vec(abs.(A) .> 0) # Find zeros from sparsification
        A_subset = y/(X[which_rows, :]) # Redo best fit
        A[which_rows] = A_subset # Update non-zero weights
        A[.!which_rows] .= 0.0 # Retain the previous zeros
    end

    return A
end

"""
Implements a hard cap on the number of terms allowed

see also: keep_n_terms
"""
function private_sparse_regression(alg::slstNumber, X::Matrix, y::Matrix,
                            which_data_row::Number, A=nothing)
    if A == nothing
        A = y/X
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


export slstHard, slstQuantile, slstNumber, denseSolver, SparseSolver,
    sparse_regression, copy_optimizer
