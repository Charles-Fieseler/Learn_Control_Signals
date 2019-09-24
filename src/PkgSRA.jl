module PkgSRA

    ## Base
    using DiffEqFlux
    using Flux
    using OrdinaryDiffEq
    ## For derivatives
    # using DSP
    # using Statistics
    ## General
    using Plots
    # using Random

    include("numerical_derivatives.jl")
    include("initialize_control_signal.jl")
    include("dmdc.jl")
    include("sindyc.jl")
    include("../utils/array_utils.jl")
    include("../utils/regression_utils.jl")
    include("../utils/control_utils.jl")
    include("forcing_functions.jl")

end # module
