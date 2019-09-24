using PkgSRA
using Plots, Random, Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
using DiffEqBayes, Distributions, Turing
using StatsPlots
using DSP
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
# include("paper_settings.jl")
include("example_lorenz.jl")

# Define the forcing functions
num_ctr = 5;
    U_starts = rand(num_ctr,1) .* tspan[2]
    U_width = 0.05;
    my_U_func_time(t, u) = U_func_time(t, u, U_width, U_starts, F_dim=3)

#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time)
dat = Array(sol)

plot3d(dat[1, :], dat[2, :], dat[3, :])
    title!("Data")

# # Interpolated function of data
# dat_interp_vec = [
#     CubicSplineInterpolation(ts, Float64.(dat[i,:])) for i in 1:size(dat,1)
# ]
# function dat_interp_func(t)
#     return t .|> dat_interp_vec
# end

# Plot to make sure this works
# dat_interp = hcat(dat_interp_func.(collect(ts))...)
# plot(vec(dat_interp[1,:]), label="interp")
#     plot!(vec(dat[1,:]), label="dat")

numerical_grad = numerical_derivative(dat, ts)
diffeq_grad = hcat(sol(ts, Val{1})...)
true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end
plot(numerical_grad[3,:], label="numerical")
    plot!(diffeq_grad[3,:], label="diffeq")
    plot!(true_grad[3,:], label="true uncontrolled")

U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time(t, dat[:,i])
end

plot(U_true')

#####
##### Define a regular Turing problem
#####

# Helper function so that ForwardDiff works
# function test_f(u, p)
#   _prob = remake(lorenz_system;u0=convert.(eltype(p),prob.u0),p=p)
#   solve(_prob,Vern9(),save_everystep=false)[end]
# end
function lorenz_system(u, p, t)
    u = convert.(eltype(p),u)

    ρ, σ, β = p
    x, y, z = u
    du = [σ*(y-x); x*(ρ-z) - y; x*y - β*z]
    return du
end

@model lorenz_grad_residual(y) = begin
    # Lorenz parameters
    ρ ~ Normal(p[1], 1.0)
    σ ~ Normal(p[2], 0.1)
    β ~ Normal(p[3], 0.1)
    params = [ρ, σ, β]

    # Observations are the residuals between model and data gradients
    # du = ones(3)
    t = [0]
    # for i in 1:length(y)
    for i in 1:size(y,2)
        u = dat[:,i]
        du_params = lorenz_system(u, params, t)
        # y[i] ~ Normal(du_params[3], 10.0)
        y[:, i] ~ MvNormal(du_params, [10.0, 10.0, 10.0])
        # y[i] = sum(abs2, du_params .- diffeq_grad[:,i])
    end
end;

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 2000
    ϵ = 0.02
    τ = 10
# Try to predict the gradient from data
num_training_pts = 500
    train_ind = 201:num_training_pts+200
    # y = diffeq_grad[3,train_ind]
    y = numerical_grad[:,train_ind]
chain = sample(lorenz_grad_residual(y),
                HMC(iterations, ϵ, τ));
plot3d(dat[1, train_ind], dat[2, train_ind], dat[3, train_ind])
    title!("Training data")

plot(chain)
println("Finished")

## THE ABOVE WORKS

# Generate test data
num_samples = 100
    param_samples = sample(chain, num_samples)
    save_ind = 3
    t = [0]
    num_test_pts = 1000
    vars = [:ρ, :σ, :β]
    all_vals = zeros(num_samples, num_test_pts)
for i in 1:num_samples
    these_params = [param_samples[v].value[i] for v in vars]
    for i2 in 1:num_test_pts
        all_vals[i, i2] = lorenz_system(dat[:,i2],
                                these_params, t)[save_ind]
    end
end

this_std = std(all_vals, dims=1)
    this_mean = mean(all_vals, dims=1)

plot(vec(numerical_grad[save_ind, 1:num_test_pts]), label="data")
    plot!(vec(this_mean), ribbon=vec(this_std), fillalpha=0.5, label="samples")
    # plot!(vec(numerical_grad[save_ind, 1:num_samples]), label="numerical")

# plot(vec(diffeq_grad[save_ind, 1:num_test_pts]) .- vec(this_mean),
#     ribbon=vec(this_std), label="residual: diffeq gradient")
#     plot!(vec(U_true[3,:]), label="True control signal")

plot(vec(numerical_grad[save_ind, 1:num_test_pts]) .- vec(this_mean),
    ribbon=vec(this_std), label="residual: numerical gradient")
    plot!(vec(U_true[3,:]), label="True control signal")




##### OLD
##### Define the DiffEqBayes problem
#####

# prob = ODEProblem((du, u, p, t)->lorenz_system(du, u, p, t,
#                             U_func_time=my_U_func_time),
#                 u0, tspan, p)
# prob = ODEProblem((du, u, p, t)->lorenz_system(du, u, p, t),
#                 u0, tspan, p)
#
# priors = [Normal(p[1], 1.0), Normal(p[2], 1.0), Normal(p[3], 1.0)]
# num_samples = 1000
# tur_sol = turing_inference(prob,Tsit5(),ts,dat,priors;
#                         num_samples=num_samples,
#                         sampler = Turing.NUTS(num_samples, 0.65))



#####
##### Reuse the turing_inference function, but stay with derivatives
#####
