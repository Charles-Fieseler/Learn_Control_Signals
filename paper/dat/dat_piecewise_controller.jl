using PkgSRA
using Plots, Random, OrdinaryDiffEq, Statistics
using Distributions, Turing, StatsPlots, DSP
using BSON: @save
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
include("../scripts/paper_settings.jl")
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

# Define the piecewise forcing function
num_jumps = 10;
    U_min = -10;
    U_max = 10
    my_U_func_time = generate_U_func_piecewise(ts, (3, length(ts)),
                                        num_jumps = num_jumps,
                                        U_min=U_min, U_max=U_max,
                                        F_dim=2)
#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time)
dat = Array(sol)
numerical_grad = numerical_derivative(dat, ts)

# Get true derivative
true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end
# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time(t, dat)
end

plot(U_true')
    title!("True control signal")

#####
##### Calculate distribution of residuals
#####
# Calculate posterior distribution of parameters
chain, train_ind = generate_chain(dat,
                        numerical_grad, lorenz_grad_residual)
# Generate test trajectories from the posterior
vars = [:ρ, :σ, :β]
sample_ind = 1:length(ts)
sample_trajectories, sample_noise = sample_posterior_grad(chain,
            dat, sample_ind, vars, lorenz_system)
# Calculate the residuals
sample_trajectory_mean = transpose(drop_all_1dims(mean(sample_trajectories, dims=1)))
    sample_trajectory_noise = mean(sample_noise)

# Align the signals
dat_grad = numerical_grad[:, sample_ind]

residual = dat_grad .- sample_trajectory_mean
ctr_guess = process_residual(residual, sample_trajectory_noise)

# plot(residual, lw=2, ribbon=mean(sample_noise), fillalpha=0.5,
#             linecolor=COLOR_DICT["intrinsic"], legend=false);
#     title!("Residual", titlefontsize=24)
#     plot!(U_true[save_ind,sample_ind][this_plot_ind], label="True controller")


#####
##### Save everything; plotted in a different script
#####

# Intermediate data
fname = DAT_FOLDERNAME*"dat_piecewise_controller_intermediate.bson";
@save fname chain sol sample_trajectories sample_noise sample_ind numerical_grad

# Original ODE and controller variables
fname = DAT_FOLDERNAME*"dat_piecewise_controller_ode_vars.bson";
@save fname dat dat_grad true_grad U_true

# Bayesian variables
fname = DAT_FOLDERNAME*"dat_piecewise_controller_ctr_vars.bson";
@save fname sample_trajectory_mean sample_trajectory_noise residual ctr_guess
