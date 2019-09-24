using Interpolations, DataStructures, Lasso, DataFrames
include("dmdc.jl");
include("../utils/combinatorics_utils.jl");
include("../utils/sindy_utils.jl")
include("../utils/regression_utils.jl")


#####
##### SINDYc model object and methods
#####
struct sindyc_model
    # After fitting
    A::Matrix
    B::Matrix
    # Interpolated Control signal
    U::Matrix
    U_func::Function
    # For augmenting data
    library::OrderedDict{Function,Any}
    # For printing equations
    variable_names
end

(m::sindyc_model)(X) = m.A*augment_data(m, X) + m.B*m.U
(m::sindyc_model)(X, t) = m.A*augment_data(m, X) + m.B*m.U_func(t)
(m::sindyc_model)(u::AbstractArray,p,t) = m(u, t) # OrdinaryDiffEq syntax

augment_data(m::sindyc_model, X) =
    calc_augmented_data(X, m.library)

intrinsic_dynamics(m::sindyc_model, X) = m.A*augment_data(m, X)
control_signal(m::sindyc_model, t) = m.B*m.U_func(t)
control_signal(m::sindyc_model) = m.B*m.U

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
##### Fitting function; effectively the constructor
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


export sindyc, calc_cross_terms, print_equations,
    build_term_names, sindyc_model, augment_data
