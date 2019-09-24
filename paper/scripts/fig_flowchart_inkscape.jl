using PkgSRA
using Plots, Random, OrdinaryDiffEq, Statistics
using Distributions, Turing
using StatsPlots, DSP
using Flux
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
# Try to predict the GRADIENT from data
num_training_pts = 500
    start_ind = 201
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
chain = sample(lorenz_grad_residual(y,train_ind), #noise, train_ind),
                NUTS(iterations, n_adapts, 0.6j_max));


#####
##### Get the residual, then control signal guess
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
for i in 1:num_samples
    these_params = [param_samples[v].value[i] for v in vars]
    all_noise[i] = param_samples[:noise].value[i]
    for (i_save, i_dat) in enumerate(train_ind)
        all_vals[i, i_save, :] = lorenz_system(dat[:,i_dat],
                                these_params, t)
    end
end

# Align the signals
dat_grad = numerical_grad[:, train_ind]

# Calculate the residuals per variable
this_std = reshape(std(all_vals, dims=1), num_test_pts, size(all_vals,3))
    this_mean = reshape(mean(all_vals, dims=1), num_test_pts, size(all_vals,3))

# Final processing
# NOTE: This is the residual for the indices: train_ind
residual = dat_grad .- transpose(this_mean)
# ctr_guess = process_residual(residual, this_std[reconstruction_plot_ind])
ctr_guess = process_residual(residual, mean(all_noise))

#####
##### Create a new model, subtracting the residual
#####

# Try to predict the MODIFIED GRADIENT from data
y = numerical_grad[:,train_ind] .- ctr_guess
# Actually sample
chain_ctr = sample(lorenz_grad_residual(y, train_ind),
                    NUTS(iterations, n_adapts, 0.6j_max));


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
# f(i) = plot!(trajectory_samples_ctr[i][1,reconstruction_plot_ind],
#             label="Trajectory_$i", color=:grey, alpha=0.8)
# plot(dat[1,train_ind], label="Data")
#     f(1)
#     f(2)


#####
##### Initialize the controller NN
#####
# Note: Only doing a subset of the time series
ctr_ts = (collect(ts)[train_ind])'
nn_dim = 128
    U_dim = 3
ctr_dyn = Chain(Dense(1,nn_dim, initb=(x)-> tspan[2].*rand(x)),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, U_dim))
ps = Flux.params(ctr_dyn)
    loss_U() = sum(abs2,ctr_dyn(ctr_ts) .- Float32.(ctr_guess))
    loss_to_use = loss_U

# Initial fast learning
tol = 500
    dat_dummy = Iterators.repeated((), 50)
    opt = ADAM(1e-3)
    num_iter = 1
    max_iter = 100
while loss_to_use() > 5*tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    global num_iter += 1
    num_iter > max_iter && break
    pt = plot(Flux.data(ctr_dyn(ctr_ts)[3,:]))
        plot!(U_true[3,train_ind])
        title!("Iteration $num_iter")
        display(pt)
end

opt = ADAM(1e-6)
    num_iter = 1
while loss_to_use() > tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    global num_iter += 1
    num_iter > max_iter && break
end
println("Finished learning control signal")



#####
##### Plot everything
#####

## 1:  3d example plot: data
plot_data = plot3d(dat[1, :], dat[2, :], dat[3, :],
        color=COLOR_DICT["true"], lw=2, legend=false,
        titlefontsize=32)
        title!("Data")

fname = FIGURE_FOLDERNAME * "fig_flowchart_inkscape_data.png";
savefig(plot_data, fname)


## 2: 3d example plot: uncontrolled
uncontrolled_ind = 1:200;
    ind = uncontrolled_ind;
plot_uncontrolled = plot3d(
        all_vals[1, ind, 1], all_vals[1, ind, 2], all_vals[1, ind, 3],
        color=COLOR_DICT["model_uncontrolled"], alpha=0.8, lw=2,
        legend=false)
    plot3d!(dat[1, ind], dat[2, ind], dat[3, ind],
            color=COLOR_DICT["true"], lw=2)
    title!("Naive Model", titlefontsize=28)

fname = FIGURE_FOLDERNAME * "fig_flowchart_inkscape_uncontrolled.png";
savefig(plot_uncontrolled, fname)


## 3: Residual with noise envelope (z coordinate)
plot_coordinate = 3;
residual_ind = 1:200;
    ind = residual_ind;
plot_residual = plot(residual[plot_coordinate,ind],
                    ribbon=mean(all_noise), fillalpha=0.5,
                    color=COLOR_DICT["residual"], legend=false,
                    xticks=false, yticks=false, lw=3)
    title!("Residual", titlefontsize=28)

fname = FIGURE_FOLDERNAME * "fig_flowchart_inkscape_residual.png";
savefig(plot_residual, fname)


## 4: Control signal guess (z coordinate)
# Same ind as above
plot_ctr_guess = plot(ctr_guess[plot_coordinate,ind],
                    color=COLOR_DICT["control_time"], legend=false,
                    xticks=false, yticks=false, lw=3)
    title!("Control Signal Guess", titlefontsize=28)

fname = FIGURE_FOLDERNAME * "fig_flowchart_inkscape_control_guess.png";
savefig(plot_ctr_guess, fname)


## 5: Controlled model
controlled_ind = 1:500;
    ind = controlled_ind;
    ctr_dat = trajectory_samples_ctr[1];
plot_controlled = plot3d(
        ctr_dat[1, ind], ctr_dat[2, ind], ctr_dat[3, ind],
        color=COLOR_DICT["model_controlled"], alpha=0.8, lw=2,
        legend=false)
    plot3d!(dat[1, ind], dat[2, ind], dat[3, ind],
            color=COLOR_DICT["true"], lw=2)
    title!("Controlled Model", titlefontsize=28)

fname = FIGURE_FOLDERNAME * "fig_flowchart_inkscape_controlled.png";
savefig(plot_controlled, fname)
