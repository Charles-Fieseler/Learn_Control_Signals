using DataStructures

#####
##### Abstract type
#####

abstract type DynamicalSystemModel end

#####
##### SINDY model object and methods
#####
struct sindyModel <: DynamicalSystemModel
    ts::Vector
    # After fitting
    A::Matrix
    # For augmenting data
    library::OrderedDict{Function,Any}
    # For printing equations
    variable_names
    # For initial training and retraining
    optimizer::SparseSolver
end

# Convenience method with a default BAD optimizer
sindyModel(ts, A, lib, var) =
    sindyModel(ts, A, lib, var, denseSolver())

(m::sindyModel)(X) = m.A*augment_data(m, X)
(m::sindyModel)(u::AbstractArray,p,t) = m(u, t) # OrdinaryDiffEq syntax

# Functions for both uncontrolled and controlled
augment_data(m::DynamicalSystemModel, X) =
    calc_augmented_data(X, m.library)



#####
##### SINDYc model object and methods
#####
struct sindycModel <: DynamicalSystemModel
    ts::Vector
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
    # For initial training and retraining
    optimizer::SparseSolver
end

# Convenience method with a default BAD optimizer
sindycModel(ts, A, B, U, Uf, lib, var) =
    sindycModel(ts, A, B, U, Uf, lib, var, denseSolver())

(m::sindycModel)(X) = m.A*augment_data(m, X) .+ m.B*m.U
(m::sindycModel)(X, t) = m.A*augment_data(m, X) .+ m.B*m.U_func(t)
(m::sindycModel)(u::AbstractArray,p,t) = m(u, t) # OrdinaryDiffEq syntax

intrinsic_dynamics(m::sindycModel, X) = m.A*augment_data(m, X)
control_signal(m::sindycModel, t) = m.B*m.U_func(t)
control_signal(m::sindycModel) = m.B*m.U


###
### Export
###
export sindyModel, sindycModel,
    augment_data, intrinsic_dynamics, control_signal
