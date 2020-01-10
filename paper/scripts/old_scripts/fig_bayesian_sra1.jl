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

# plot3d(dat[1, :], dat[2, :], dat[3, :])
#     title!("Data")

# Derivatives
numerical_grad = numerical_derivative(dat, ts)
true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end
# plot(numerical_grad[3,:], label="numerical")
#     plot!(true_grad[3,:], label="true uncontrolled")

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time(t, dat[:,i])
end

# plot(U_true')

#####
##### Define the Bayesian modeling framework
#####

# Define the model
function lorenz_system(u, p, t)
    u = convert.(eltype(p),u)

    ρ, σ, β = p
    x, y, z = u
    du = [σ*(y-x); x*(ρ-z) - y; x*y - β*z]
    return du
end

@model lorenz_grad_residual(y, noise, train_ind) = begin
    # Lorenz parameters
    ρ ~ Normal(p[1], 1.0)
    σ ~ Normal(p[2], 0.1)
    β ~ Normal(p[3], 0.1)
    params = [ρ, σ, β]

    t = [0] # Not time dependent
    for i in 1:size(y,2)
        u = dat[:,train_ind[i]]
        du_params = lorenz_system(u, params, t)
        # TODO: Hard-coded noise
        #   Note: will not converge if much lower...
        y[:, i] ~ MvNormal(du_params, [noise, noise, noise])
    end
end;

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 200
    ϵ = 0.01
    τ = 10
    noise = 5.0
# Try to predict the GRADIENT from data
num_training_pts = 500
    start_ind = 1
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
chain = sample(lorenz_grad_residual(y, noise, train_ind),
                HMC(iterations, ϵ, τ));
# plot3d(dat[1, train_ind], dat[2, train_ind], dat[3, train_ind])
#     title!("Training data")
# plot(chain)

# Generate test trajectories from the posterior
num_samples = 100
    param_samples = sample(chain, num_samples)
    save_ind = 3
    t = [0]
    num_test_pts = 500
    vars = [:ρ, :σ, :β]
    all_vals = zeros(num_samples, num_test_pts)
for i in 1:num_samples
    these_params = [param_samples[v].value[i] for v in vars]
    for (i_save, i_dat) in enumerate(train_ind)
        all_vals[i, i_save] = lorenz_system(dat[:,i_dat],
                                these_params, t)[save_ind]
    end
end

# Calculate the residuals
this_std = std(all_vals, dims=1)
    this_mean = mean(all_vals, dims=1)

# Align the signals
dat_grad = numerical_grad[save_ind, train_ind]

# Final processing
this_plot_ind = 1:300
    this_ts = collect(ts)[this_plot_ind]
    
residual = vec(dat_grad[this_plot_ind]) .- vec(this_mean[this_plot_ind])
ctr_guess = process_residual(residual, this_std[this_plot_ind])

#####
##### Plot everything
#####

# p0 = plot3d(dat[1, :], dat[2, :], dat[3, :]);
#     title!("Data");

p1 = plot(vec(dat_grad[this_plot_ind]),
            linecolor=COLOR_DICT["true"], label="Data", lw=3);
    plot!(vec(this_mean[this_plot_ind]),
            linecolor=COLOR_DICT["intrinsic"], ribbon=vec(this_std),
            fillalpha=0.5, label="Models", lw=1, xtick=false,
            legendfontsize=12);
    # plot!(vec(true_grad[3,plot_ind][this_plot_ind]), label="Truth")
    title!("Reconstruction: No Noise", titlefontsize=24);

p2 = plot(residual, lw=2, ribbon=vec(this_std), fillalpha=0.5,
            linecolor=COLOR_DICT["intrinsic"], legend=false);
    title!("Residual", titlefontsize=24);

p3 = plot(this_ts, ctr_guess, lw=2, label="Processed Residual",
            linecolor=COLOR_DICT["control_time"]);
    plot!(this_ts, vec(U_true[3,plot_ind][this_plot_ind]), label="True control signal",
                linecolor=COLOR_DICT["control_true"],
                legendfontsize=12);
    xlabel!("Time")
    title!("Learned signal", titlefontsize=24);

# All together
l = @layout [p1; p2; p3]
    p_all = plot(p1, p2, p3, layout=l)
fname = FIGURE_FOLDERNAME * "fig_bayesian_sra1.png";
# savefig(p_all, fname)
