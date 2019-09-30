using PkgSRA
using Plots, Random, OrdinaryDiffEq, Statistics
using Distributions, Turing
using StatsPlots, DSP
using Flux
using BSON: @save
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("../scripts/paper_settings.jl")
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

# Define the multivariate forcing function
num_ctr = 50;
    U_starts = rand(3, num_ctr) .* tspan[2]
    U_widths = [0.1, 0.1, 0.1];
    amplitudes = [200.0, 200.0, 200.0]
my_U_func_time2(t, u) = U_func_time_multivariate(t, u,
                        U_widths, U_starts,
                        F_dim=[1, 2, 3],
                        amplitudes=amplitudes)

#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time2)
dat = Array(sol)

# Derivatives
numerical_grad = numerical_derivative(dat, ts)

true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t, dat[:,i])
end

#####
##### Calculate distribution of residuals
#####
# Calculate posterior distribution of parameters
chain, train_ind = generate_chain(dat,
                        numerical_grad, lorenz_grad_residual)
# Generate samples of the gradient time series from the posterior
vars = [:ρ, :σ, :β]
sample_ind = 1:length(ts)
sample_gradients, sample_noise = sample_posterior_grad(chain,
            dat, sample_ind, vars, lorenz_system)
# Generate test trajectories from the posterior
sample_trajectories = sample_posterior_trajectories(
                                chain, dat, sample_ind, vars, lorenz_system,
                                tspan, ts, num_samples=5)[1]
# Calculate the residuals
sample_gradients = transpose(drop_all_1dims(mean(sample_gradients, dims=1)))
    sample_trajectory_noise = mean(sample_noise)

let y = sample_trajectories[3], y2 = sample_trajectories[4]
    plot3d(y[1,:], y[2,:], y[3,:], label="Naive Sample 2")
    plot3d!(y2[1,:], y2[2,:], y2[3,:], label="Naive Sample 1")
    plot3d!(dat[1,:], dat[2,:], dat[3,:], label="Data")
end

# Align the signals
dat_grad = numerical_grad[:, sample_ind]

residual = dat_grad .- sample_trajectory_mean
ctr_guess = process_residual(residual, sample_trajectory_noise)

# Visualization
i = 3
    # ind = 1:300
    # plot(residual[i,ind], ribbon=sample_trajectory_noise, label="Residual")
    plot(ctr_guess[i,:], label="Control guess")
    plot!(U_true[i,:], label="True", show=true)

#####
##### Create a new model, subtracting the residual
#####
# Try to predict the MODIFIED GRADIENT from data
y = numerical_grad[:,sample_ind] .- ctr_guess
chain_ctr, train_ind = generate_chain(dat, y, lorenz_grad_residual)
density(chain_ctr)
# Generate test trajectories from the CORRECTED posterior
sample_trajectories_ctr, sample_noise_ctr = sample_posterior_trajectories(
                                chain_ctr, dat, sample_ind, vars, lorenz_system,
                                tspan, ts, num_samples=5)
# Calculate the residuals
sample_trajectory_noise_ctr = mean(sample_noise_ctr)

let y = sample_trajectories_ctr[1], y2 = sample_trajectories_ctr[2]
    plot3d(y[1,:], y[2,:], y[3,:], label="Corrected Sample 1")
    plot3d!(y2[1,:], y2[2,:], y2[3,:], label="Corrected Sample 2")
    plot3d!(dat[1,:], dat[2,:], dat[3,:], label="Data")
end

#####
##### Initialize the controller NN
#####
# NOTE: doesn't really work to learn these spikes in time...
# Note: Only doing a subset of the time series
# nn_dim = 128
#     U_dim = 3
# ctr_dyn = Chain(Dense(1,nn_dim, initb=(x)-> tspan[2].*rand(x)),
#                 Dense(nn_dim, nn_dim, σ),
#                 Dense(nn_dim, nn_dim, σ),
#                 Dense(nn_dim, U_dim))
# ctr_ts = (collect(ts)[train_ind])'
# ps = Flux.params(ctr_dyn)
#     loss_U() = sum(abs2,ctr_dyn(ctr_ts) .- Float32.(ctr_guess))
#     loss_to_use = loss_U
#
# # Initial fast learning
# tol = 500
# train_in_loop(ps, tol, loss_to_use, rate=1e-2, max_iter=3)
# train_in_loop(ps, tol, loss_to_use, rate=1e-6, max_iter=3)
# println("Finished learning control signal")


#####
##### Save everything; plotted in a different script
#####
this_dat_name = DAT_FOLDERNAME*"dat_flowchart_inkscape_"

# Intermediate data
fname = this_dat_name*"intermediate.bson";
@save fname chain chain_ctr sol sample_trajectories sample_trajectories_ctr sample_noise sample_noise_ctr sample_ind numerical_grad

# Original ODE and controller variables
fname = this_dat_name*"ode_vars.bson";
@save fname dat dat_grad true_grad U_true

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars.bson";
@save fname sample_trajectory_mean sample_trajectory_noise residual ctr_guess

# Bayesian variables 1: after control
fname = this_dat_name*"ctr_vars.bson";
@save fname sample_trajectories_ctr sample_trajectory_noise_ctr

# TODO: NN variables
