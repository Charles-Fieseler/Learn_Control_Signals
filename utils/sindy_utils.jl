using DataStructures

#####
##### Helper functions
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

 # Convert library strings to functions
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
    build_dataframe

# function print_equations(model::sindyc_model;
#                          var_names=nothing, tol=1e-2, digits=2)
#     n, m = size(model.A)
#     if var_names == nothing
#         default_names = ["x"; "y"; "z"]
#         var_names = default_names[1:n]
#     end
#     library_vec = collect(model.library)
#     for i_var in 1:n
#         this_var = var_names[i_var]
#         print("d$this_var/dt = ")
#         these_term_names = copy(var_names)
#         library_index = 1
#         first_term = true
#         for i_term in 1:m
#             val = round(model.A[i_var, i_term], digits=digits)
#             if abs(val) > tol
#                 this_term = these_term_names[1]
#                 if first_term
#                     print("$val$this_term")
#                     first_term = false
#                 else
#                     print(" + $val$this_term")
#                 end
#             end
#             if i_term < m
#                 if length(these_term_names) == 1
#                     next_library_func, args = library_vec[library_index]
#                     these_term_names = vec(next_library_func(var_names, args))
#                     library_index += 1
#                 else
#                     deleteat!(these_term_names, 1)
#                 end
#             else
#                 println("\b\b")
#             end
#         end
#     end
# end
