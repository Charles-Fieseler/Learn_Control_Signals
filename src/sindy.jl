using DataStructures

#####
##### Abstract type
#####

abstract type DynamicalSystemModel end

#####
##### SINDY model object and methods
#####
mutable struct sindyModel <: DynamicalSystemModel
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
# Initializer for creating templates
z = zeros(1,1)
sindyModel(lib, var, opt::SparseSolver) =
    sindyModel([0], z, lib, var, opt)

(m::sindyModel)(X) = m.A*augment_data(m, X)
(m::sindyModel)(X, t) = m.A*augment_data(m, X) # To match syntax below
(m::sindyModel)(u::AbstractArray,p,t) = m(u, t) # OrdinaryDiffEq syntax

# Functions for both uncontrolled and controlled
augment_data(m::DynamicalSystemModel, X) =
    calc_augmented_data(X, m.library)



#####
##### SINDYc model object and methods
#####
mutable struct sindycModel <: DynamicalSystemModel
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
# Initializer for creating templates
z = zeros(1,1)
sindycModel(lib, var, opt::SparseSolver) =
    sindycModel([0], z, z, z, ()->0, lib, var, opt)

(m::sindycModel)(X) = m.A*augment_data(m, X) .+ m.B*m.U
(m::sindycModel)(X, t) = m.A*augment_data(m, X) .+ m.B*m.U_func(t)
(m::sindycModel)(u::AbstractArray,p,t) = m(u, t) # OrdinaryDiffEq syntax

intrinsic_dynamics(m::sindycModel, X) = m.A*augment_data(m, X)
control_signal(m::sindycModel, t) = m.B*m.U_func(t)
control_signal(m::sindycModel) = m.B*m.U


# Basic integration functions
function simulate_model(m::DynamicalSystemModel, u0)
    condition(u,t,integrator) = any(abs.(u).>1e4)
    cb = DiscreteCallback(condition, terminate!)
    prob = ODEProblem(m, u0, m.ts, [0], callback=cb)
    return Array(solve(prob, Tsit5(), saveat=m.ts));
end


###
### Export
###
export sindyModel, sindycModel,
    augment_data, intrinsic_dynamics, control_signal,
    simulate_model
