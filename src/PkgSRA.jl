module PkgSRA

    ## Base
    using DiffEqFlux, OrdinaryDiffEq, Flux, Turing, Random
    ## For derivatives
    using DSP, Statistics
    ## General
    using Plots, StatsPlots, BSON
    using DataStructures

    include("numerical_derivatives.jl")
    include("sindyc.jl")
    include("dmdc.jl")
    include("initialize_control_signal.jl")
    include("../utils/array_utils.jl")
    include("../utils/regression_utils.jl")
    include("../utils/control_utils.jl")
    include("../utils/posterior_sampling_utils.jl")
    include("forcing_functions.jl")

end # module
