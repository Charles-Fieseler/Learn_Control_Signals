using Interpolations, DataStructures, Lasso, DataFrames, Flux
include("../src/sindyc.jl")
include("../src/dmdc.jl");
include("../utils/combinatorics_utils.jl");
include("../utils/regression_utils.jl")

#####
##### Fitting function; effectively the default constructor
#####
"""
Naive implementation of Sparse Identification of Nonlinear DYnamics
with control (SINDYc). To actually use sparsity, the Lasso.jl package
is required

There are several nonlinear library terms that can be implemented via passing
a list of strings and arguments via 'library', but a custom function can also be
passed using 'custom_func'. Currently implemented library terms are:
    ["cross_terms", order::Int]
        Here, 'order' is how high of an order to do
"""
function sindyc(X, X_grad=nothing, U=nothing, ts=nothing;
                library=Dict(),
                use_lasso=false,
                hard_threshold=nothing,
                quantile_threshold=0.1,
                var_names = ["x", "y", "z"])
                #TODO: implement custom functions
    if X_grad == nothing
        X_grad = numerical_derivative(X)
    end
    if ts == nothing
        ts = 1:size(X,1)
    end

    # Get a model with the augmented data
    n, m = size(X)
    library = convert_string2function(library)
    X_augmented = calc_augmented_data(X, library)
    if !use_lasso
        A, B = dmdc(X_augmented, X_grad, U)
        A = A[1:n, :]
        B = B[1:n, :]
    else
        # UPDATE: use my own sequential least squares threshold
        # TODO: make work with control signal
        if U == nothing
            A = sparse_regression(X_augmented, X_grad,
                                hard_threshold=hard_threshold,
                                quantile_threshold=quantile_threshold)
        else
            Ω = [X_augmented; U]
            AB = sparse_regression(Ω, X_grad,
                                hard_threshold=hard_threshold,
                                quantile_threshold=quantile_threshold)
            A = AB[:, 1:size(X_augmented,1)]
            B = AB[:, size(X_augmented,1)+1:end]
        end

        # Do lasso with cross validation; loop over variables
        # A = ones(n, size(X_augmented,1))
        # df = build_dataframe(X_augmented, X_grad, library)
        # regressors = names(df)[n+1:end]
        # for i in 1:n
            #TODO: the formula here may need to be updated
            # this_predictor = names(df)[i] # Derivatives
            # this_f = this_predictor ~ sum(StatsModels.terms.(regressors))
            # lasso_model = fit(LassoModel, this_f, df;
            #                 select=MinCVmse(Kfold(3,2)))
        # end
    end


    if U == nothing
        model = sindyc_model(A, zeros(n,1), zeros(1, m), (t)->zeros(1),
                            library, var_names)
    else
        # TODO: make U_func work with multiple channels
        U_func(t) = CubicSplineInterpolation(ts, vec(U))(t)
        model = sindyc_model(A, B, U, U_func,
                            library, var_names )
    end

    return model
end

#####
##### Helper functions for pretty printing
#####
"""
Prints equations of a SINDy model using saved or passed variable names
    Note: if 0.0 is displayed, this means the coefficient was nonzero to the
    tolerance specified in 'tol' but below the rounding threshold
"""
function print_equations(model::sindyc_model;
                         var_names=nothing, tol=1e-4, digits=1)
    n, m = size(model.A)
    if var_names == nothing
        default_names = ["x"; "y"; "z"]
        var_names = default_names[1:n]
    end
    term_names = build_term_names(model, var_names)
    for i_var in 1:n
        this_var = var_names[i_var]
        print("d$this_var/dt = ")
        first_term = true
        for i_term in 1:m
            val = model.A[i_var, i_term]
            if abs(val) > tol
                this_term = term_names[i_term]
                val = round(val, digits=digits)
                if first_term
                    print("$val$this_term")
                    first_term = false
                else
                    print(" + $val$this_term")
                end
            end
        end
        println("\b\b")
    end
end

#####
##### Helper functions for nonlinear terms
#####
"""
Calculates higher powers of individual rows of X, but not cross terms
"""
function calc_power_terms(X, order)
    n = size(X, 1)
    all_powers = nchoosek(order, n)
    popfirst!(all_powers) # i.e. remove the identity
    power_terms = nothing

    for p_list in all_powers
        this_power_term = zeros(size(X))
        for (i,p) in enumerate(p_list)
            this_power_term[i,:] = X[i,:] .^ p
        end
        if power_terms == nothing
            power_terms = copy(this_power_term)
        else
            power_terms = vcat(power_terms, this_power_term)
        end
    end

    return power_terms
end

"""
Calculates higher powers of individual rows of X, AND cross terms
"""
function calc_cross_terms(X, order)
    if ndims(X) == 1
        X = reshape(X, (length(X), 1))
    end
    n, m = size(X)
    all_powers = calc_permutations(n, order)
    cross_terms = nothing

    for p_list in all_powers
        this_cross_term = ones(eltype(X), 1, m)
        for p in p_list
            # Keep the X slice 2 dimensional
            this_cross_term .*= X[p:p,:]
        end
        if cross_terms == nothing
            cross_terms = copy(this_cross_term)
        else
            cross_terms = vcat(cross_terms, this_cross_term)
        end
    end

    return cross_terms
end

"""
Returns a constant; has the right function signature
"""
function calc_constant(X, tmp)
    return ones(eltype(X), 1, size(X,2))
end

FUNCTION_DICT = Dict("cross_terms"=>calc_cross_terms,
                     "power_terms"=>calc_power_terms,
                     "constant"=>calc_constant)

#####
##### Helper functions for variable names
#####
 """Convert library strings to functions"""
 function convert_string2function(library)
     func_library = OrderedDict{Function, Any}()
     for f in keys(library)
         if f isa String
             f_key = FUNCTION_DICT[f]
         else
             f_key = f
         end
         func_library[f_key] = library[f] # Key is now the function
     end
     return func_library
 end


 function build_term_names(library::Union{Dict, OrderedDict}, var_names)
     term_names = copy(var_names)
     # Now we pass through library functions
     library_vec = collect(library)
     for (f, args) in library_vec
         these_term_names = vec(f(var_names, args))
         term_names = vcat(term_names, these_term_names)
     end

     return term_names
 end
 build_term_names(model::sindyc_model, var_names) =
     build_term_names(model.library, var_names)
 build_term_names(model::sindyc_model) =
     build_term_names(model.library, model.variable_names)


 #####
 ##### Helper functions for calculation
 #####
"""Uses saved library functions to calculate the input space"""
function calc_augmented_data(X, library)
    # Apply functions and save
    X_augmented = copy(X)
    for (f, args) in pairs(library)
        X_augmented = vcat(X_augmented, f(X, args))
    end

    n, m = size(X_augmented)
    if m == 1
        # ode solvers want a vector
        X_augmented = reshape(X_augmented, (n))
    end

    return X_augmented
end

# Flux version!
#   Vertical concatenation doesn't easily work
function calc_augmented_data(X::TrackedArray, library)
    # Apply functions and save
    X_augmented = copy(X)
    for (f, args) in pairs(library)
        X_augmented = vcat(X_augmented,
                Flux.data.(f(X, args)))
    end

    n, m = size(X_augmented)
    if m == 1
        # ode solvers want a vector
        X_augmented = reshape(X_augmented, (n))
    end

    return X_augmented
end

"""
Build dataframe from the augmented data
"""
function build_dataframe(X_augmented, dXdt, library)
    var_names = ["x", "y", "z"]
    df = DataFrame()
    # First add derivatives
    for i in 1:size(dXdt,1)
        name = var_names[i]
        name_sym = Symbol("d$name/dt")
        df[!, name_sym] = vec(dXdt[i,:])
    end
    # Now other terms
    term_names = build_term_names(library, var_names)
    for i in 1:length(term_names)
        name = term_names[i]
        if length(name) == 0
            name = "const"
        end
        name_sym = Symbol(name)
        df[!, name_sym] = vec(X_augmented[i,:])
    end

    return df
end


export calc_augmented_data, convert_string2function, calc_constant,
    FUNCTION_DICT, calc_cross_terms, calc_power_terms,
    build_dataframe, sindyc, print_equations
