module PkgSRA

    ## Base
    using DiffEqFlux, OrdinaryDiffEq, Flux, Turing, Random
    ## For derivatives
    using DSP, Statistics
    ## General
    using Plots, StatsPlots, BSON
    using DataStructures

    # Core numerical algorithms with control
    include("numerical_derivatives.jl")
    include("sindyc.jl")
    include("dmdc.jl")
    include("initialize_control_signal.jl")
    include("forcing_functions.jl")

    # Utilities
    include("../utils/array_utils.jl")
    include("../utils/sindy_utils.jl")
    include("../utils/regression_utils.jl")
    include("../utils/control_utils.jl")
    include("../utils/posterior_sampling_utils.jl")
    include("../utils/combinatorics_utils.jl");

    # State of the iterative algorithm, and plotting utilities
    include("../utils/main_algorithm_utils.jl")
    include("../src/sra_model_object.jl");
    include("../src/sra_model_functions.jl");
    include("../src/sra_model_plotting.jl");

end # module
