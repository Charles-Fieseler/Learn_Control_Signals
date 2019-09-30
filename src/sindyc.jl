using DataStructures
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

export sindyc_model
