using PkgSRA, Test, Random
Random.seed!(13)

###
### First, Lorenz
###
include("../examples/example_lorenz.jl")
dat = Array(solve_lorenz_system())
true_grad = core_dyn_true(dat)

# Numerical derivative: the main function to be tested
numerical_grad = numerical_derivative(dat, ts)

# Basic
@test !isempty(numerical_grad)
@test size(numerical_grad) == size(true_grad)

# Accuracy
@test isapprox(numerical_grad, true_grad, rtol=1e-2)
@test !isapprox(numerical_grad, true_grad, rtol=1e-3)


###
### Second, SIR
###
include("../examples/example_sir.jl")
dat = Array(solve_sir_system())
true_grad = core_dyn_true(dat)

# Numerical derivative: the main function to be tested
numerical_grad = numerical_derivative(dat, ts)

# Basic
@test !isempty(numerical_grad)
@test size(numerical_grad) == size(true_grad)

# Accuracy
@test isapprox(numerical_grad, true_grad, rtol=1e-2)
@test !isapprox(numerical_grad, true_grad, rtol=1e-3)

# Check why the above doesn't work...
plot(numerical_grad[1,:], label="numerical");
    plot!(true_grad[1,:], label="true")

dt = ts[2] - ts[1]
bad_grad = (dat[:,2:end] .- dat[:,1:end-1]) / dt
plot(ts, numerical_grad[1,:], label="numerical");
    plot!(ts, bad_grad[1,:], label="one-step")
    plot!(ts, true_grad[1,:], label="true")
