using PkgSRA
using Plots, Random, OrdinaryDiffEq, Statistics
using Distributions, Turing
using StatsPlots, DSP
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("paper_settings.jl")
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

# Define the multivariate forcing function
num_ctr = 50;
    U_starts = rand(3, num_ctr) .* tspan[2]
    U_widths = [0.05, 0.05, 0.05];
    amplitudes = [30.0, 30.0, 30.0]
my_U_func_time(t, u) = U_func_time_multivariate(t, u,
                        U_widths, U_starts,
                        F_dim=[1, 2, 3],
                        amplitudes=amplitudes)

#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time)
dat = Array(sol)

# plot3d(dat[1, :], dat[2, :], dat[3, :])
#     title!("Data")

# Derivatives
numerical_grad = numerical_derivative(dat, ts)

true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time(t, dat[:,i])
end

# plot(U_true')

#####
##### Define the Bayesian modeling framework
#####

@model lorenz_grad_residual(y, ind) = begin
    # Lorenz parameters
    ρ ~ Normal(30, 10.0)
    σ ~ Normal(10, 10.0)
    β ~ Normal(5, 5.0)
    noise ~ Truncated(Normal(10, 5.0), 0, 30)

    for i in 1:size(y,2)
        x1 = dat[1,ind[i]]
        x2 = dat[2,ind[i]]
        x3 = dat[3,ind[i]]

        y[1, i] ~ Normal(σ*(x2-x1), noise)
        y[2, i] ~ Normal(x1*(ρ-x3) - x2, noise)
        y[3, i] ~ Normal(x1*x2 - β*x3, noise)
    end
end;

# Better sampler: NUTS
iterations = 10000
    n_adapts = Int(iterations/5)
    j_max = 1.0
    # noise = 10.0
# Try to predict the GRADIENT from data
num_training_pts = 500
    start_ind = 201
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
chain = sample(lorenz_grad_residual(y,train_ind), #noise, train_ind),
                NUTS(iterations, n_adapts, 0.6j_max));
# plot3d(dat[1, train_ind], dat[2, train_ind], dat[3, train_ind])
#     title!("Training data")

# Settings for both density plots
subplot_order = [3, 4, 2] # The chain has a different variable order

density_plot = density(chain);
# for i in 1:length(p)
#     vline!([p[i]], subplot=subplot_order[i], label="True", lw=3)
# end
# TODO: Why doesn't the for loop work? Is it a scope issue?
add_true_parameter_to_subplot(i) =
        vline!([p[i]], subplot=subplot_order[i], label="True", lw=3)
    i= 1; add_true_parameter_to_subplot(i)
    i= 2; add_true_parameter_to_subplot(i)
    i= 3; add_true_parameter_to_subplot(i)
# xlims!([2.5, 3.0], subplot=2)

# Generate test trajectories from the posterior
# num_samples = 50
#     trajectory_samples = []
#     param_samples = sample(chain, num_samples)
#     t = [0]
#     vars = [:ρ, :σ, :β]
#     start_ind = 1
# for i in 1:num_samples
#     these_params = [param_samples[v].value[i] for v in vars]
#     prob = ODEProblem(lorenz_system, dat[:,start_ind], tspan, these_params)
#     sol = solve(prob, Tsit5(), saveat=ts);
#     push!(trajectory_samples, Array(sol))
# end

# Plot example trajectories for the incorrect parameter values
# reconstruction_plot_ind = 1:500
#     original_plot_ind = start_ind .+ reconstruction_plot_ind .- 1
# f(i) = plot!(trajectory_samples[i][1,reconstruction_plot_ind],
#             label="Trajectory_$i", color=:grey, alpha=0.8)
# plot(dat[1,original_plot_ind], label="Data")
#     f(1)
#     f(2)

#####
##### Get the residual, i.e. control signal guess
#####

# Generate test gradient predictions from the posterior
num_samples = 100
    param_samples = sample(chain, num_samples)
    save_ind = 3
    t = [0]
    num_test_pts = 500
    vars = [:ρ, :σ, :β]
    all_vals = zeros(num_samples, num_test_pts, size(dat,1))
    all_noise = zeros(num_samples)
    plot_ind = 201:num_test_pts+200
for i in 1:num_samples
    these_params = [param_samples[v].value[i] for v in vars]
    all_noise[i] = param_samples[:noise].value[i]
    for (i_save, i_dat) in enumerate(plot_ind)
        all_vals[i, i_save, :] = lorenz_system(dat[:,i_dat],
                                these_params, t)
    end
end

# Align the signals
dat_grad = numerical_grad[:, plot_ind]

# Calculate the residuals per variable
this_std = reshape(std(all_vals, dims=1), num_test_pts, size(all_vals,3))
    this_mean = reshape(mean(all_vals, dims=1), num_test_pts, size(all_vals,3))

this_plot_ind = 1:300
plot_coordinate = 3
this_plot = plot(vec(dat_grad[plot_coordinate, this_plot_ind]),
            linecolor=COLOR_DICT["true"], label="Training Data", lw=3);
    plot!(vec(this_mean[this_plot_ind, plot_coordinate]),
            linecolor=COLOR_DICT["intrinsic"], ribbon=vec(all_noise),
            fillalpha=0.5, label="Models", lw=1);
    plot!(vec(true_grad[plot_coordinate,plot_ind][this_plot_ind]), label="Truth")
    title!("Reconstruction of coordinate $save_ind")


# Function for processing the residual into a true guess
function process_residual(mean, std)
    low = mean .- std
    high = mean .+ std
    real_ind = .!((low .< 0.0) .& (high .> 0.0))

    ctr = zeros(size(mean))
    ctr[real_ind] = mean[real_ind]

    return ctr
end

# Final processing
# NOTE: This is the residual for the indices: plot_ind
residual = dat_grad .- transpose(this_mean)
# ctr_guess = process_residual(residual, this_std[reconstruction_plot_ind])
ctr_guess = process_residual(residual, mean(all_noise))

plot(residual[plot_coordinate,:], label="Residual", ribbon=mean(all_noise), fillalpha=0.5)
    plot!(ctr_guess[plot_coordinate,:], label="Control guess", lw=2)
    plot!(U_true[plot_coordinate,train_ind], label="Control True")

#####
##### Create a new model, subtracting the residual
#####

# Try to predict the MODIFIED GRADIENT from data
y = numerical_grad[:,train_ind] .- ctr_guess
    # y[save_ind,:] .-= ctr_guess
    # y[save_ind,:] .-= U_true[3, train_ind] # Cheating!

plot(numerical_grad[plot_coordinate,train_ind], label="Raw Gradient");
    # plot!(numerical_grad[plot_coordinate,train_ind] .- residual, label="Processed Gradient (numerical)");
    plot!(y[plot_coordinate,:], label="Processed Gradient");
    plot!(U_true[plot_coordinate,train_ind].-200, label="Controller (ideal)")
    plot!(ctr_guess[plot_coordinate,:].-200, label="Controller (numerical)")

# Actually sample
chain_ctr = sample(lorenz_grad_residual(y, train_ind),
                    NUTS(iterations, n_adapts, 0.6j_max));


density_plot = density(chain_ctr)
    i= 1; add_true_parameter_to_subplot(i)
    i= 2; add_true_parameter_to_subplot(i)
    i= 3; add_true_parameter_to_subplot(i)

# Generate test trajectories from the CORRECTED posterior
num_samples = 10
    trajectory_samples_ctr = []
    param_samples_ctr = sample(chain_ctr, num_samples)
for i in 1:num_samples
    these_params = [param_samples_ctr[v].value[i] for v in vars]
    prob = ODEProblem(lorenz_system, dat[:,start_ind], tspan, these_params)
    sol = solve(prob, Tsit5(), saveat=ts);
    push!(trajectory_samples_ctr, Array(sol))
end

# Plot example trajectories for the CORRECTED parameter values
f(i) = plot!(trajectory_samples_ctr[i][1,reconstruction_plot_ind],
            label="Trajectory_$i", color=:grey, alpha=0.8)
plot(dat[1,original_plot_ind], label="Data")
    f(1)
    f(2)


#####
##### Plot everything
#####

# TODO: Get legends working
# TODO: Increase line width
# TODO: Make sure titles don't overlap
p_all = density(chain);
    density!(chain_ctr);
    # density_plot.subplots[2][:legend] = :best
    i= 1; add_true_parameter_to_subplot(i);
    i= 2; add_true_parameter_to_subplot(i);
    i= 3; add_true_parameter_to_subplot(i)
    # legend()
    p_all

# All together
# l = @layout [p1; p2; p3]
#     p_all = plot(p1, p2, p3, layout=l)
fname = FIGURE_FOLDERNAME * "fig_parameter_distributions.png";
savefig(p_all, fname)
