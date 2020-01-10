using PkgSRA
using Plots, Random, Flux, Turing, StatsPlots
pyplot()
Random.seed!(11)
include("paper_settings.jl");

#####
##### Define the controlled dynamical system
#####
# Load example problem
include(EXAMPLE_FOLDERNAME*"example_falling_ball.jl")
# Spatial forcing function
U_starts = [6.0]
    U_widths = 0.2;
    amplitude = 50.0;
U_func_kick(t, u) = U_func_time(t, u,
                                U_widths, U_starts,
                                F_dim=2,
                                amplitude=amplitude)
# Define the time dependent forcing function
U_func_wall(X) = U_func_spring(X, k=1e3, r=1.0);

# Solve the system
dyn_with_ctr = solve_ball_system(U_func_space=U_func_wall,
                                 U_func_time=U_func_kick)
# plot(dyn_with_ctr)
dat = Array(dyn_with_ctr)

# Get the true control signals
dyn_control_wall = zeros(size(dat))
for i in 1:size(dat, 2)
    dyn_control_wall[:,i] = U_func_wall(dat[:, i])
end

dyn_control_kick = zeros(size(dat))
for (i, t) in enumerate(ts)
    dyn_control_kick[:,i] = U_func_kick(t, dat[:, i])
end

# Take numerical gradient
numerical_grad = numerical_derivative(dat, ts)

#####
##### Bayesian residual process and control guess
#####

# Define model and sample
@model ball_residual(y, ind) = begin
    # Lorenz parameters
    g ~ Normal(10, 1.0)
    noise ~ Truncated(Normal(1, 1.0), 0, 20)

    t_vec = collect(ts)
    for i in train_ind
        du = ball_system(dat[:,i], g, t_vec[i])

        # y ~ du .+ MvNormal(length(du), noise)
        y[1,i] ~ Normal(du[1], noise)
        y[2,i] ~ Normal(du[2], noise)
    end
end;

# Better sampler: NUTS
iterations = 1000
    n_adapts = Int(iterations/5)
    j_max = 1.0
# Try to predict the GRADIENT from data
num_training_pts = 1000
    start_ind = 1
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
chain = sample(ball_residual(y,train_ind), #noise, train_ind),
                NUTS(iterations, n_adapts, 0.6j_max));

plot(chain)

# Get a distribution of predictions
num_samples = 100
    param_samples = sample(chain, num_samples)
    vars = :g
    all_vals = zeros(num_samples, length(train_ind), size(dat,1))
    all_noise = zeros(num_samples)
for i in 1:num_samples
    these_params = param_samples[vars].value[i]
    all_noise[i] = param_samples[:noise].value[i]
    for (i_save, i_dat) in enumerate(train_ind)
        all_vals[i, i_save, :] = ball_system(dat[:,i_dat],
                                these_params, collect(ts)[i_dat])
    end
end

# Get the residuals
dat_grad = numerical_grad[:, train_ind]

# Calculate the residuals per variable
this_std = reshape(std(all_vals, dims=1), length(train_ind), size(all_vals,3))
    this_mean = reshape(mean(all_vals, dims=1), length(train_ind), size(all_vals,3))

# Final processing
# NOTE: This is the residual for the indices: train_ind
residual = dat_grad .- transpose(this_mean)
# ctr_guess = process_residual(residual, this_std[reconstruction_plot_ind])
ctr_guess = process_residual(residual, mean(all_noise))

plot(residual[2,:], ribbon=mean(all_noise))

plot(ctr_guess[2,:], label="Control Guess", lw=2)
    plot!(dyn_control_wall[2,:], label="Control Wall")
    plot!(dyn_control_kick[2,:], label="Control Kick")


#####
##### Fit first neural network: g(x)
#####
nn_dim = 16
    U_dim = 1 # Expect that the true dimensionality is 1
# Note: need to initialize the first layer so that the whole space is used
ctr_dyn = Chain(Dense(2,nn_dim, initb=(x)-> tspan[2].*rand(x)),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, U_dim),
                Dense(U_dim,2)) # Equivalent to the B matrix
ps = Flux.params(ctr_dyn)
    loss_U() = sum(abs2,ctr_dyn(dat) .- ctr_guess)
    loss_to_use = loss_U

# Initial fast learning
tol = 300
    dat_dummy = Iterators.repeated((), 100)
    opt = ADAM(1e-2)
    num_iter = 1
    max_iter = 5
while loss_to_use() > 5*tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    global num_iter += 1
    num_iter > max_iter && break
end

plot(Flux.data(ctr_dyn(dat)[2,:]), label="g(x)", lw=2)
    plot!(dyn_control_wall[2,:], label="Control Wall")
    plot!(dyn_control_kick[2,:], label="Control Kick")

# plot(Flux.data(ctr_dyn(dat)[1,:]), label="g(x)", lw=2)
#     plot!(dyn_control_wall[1,:], label="Control Wall")
#     plot!(dyn_control_kick[1,:], label="Control Kick")

opt = ADAM(1e-5)
    num_iter = 1
while loss_to_use() > tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    global num_iter += 1
    num_iter > max_iter && break
end
println("Finished learning control signal")












#####
##### Produce the plots
#####
# First: intrinsic dynamics
plot_data = plot(ts, dyn_with_ctr[1,:], label="Height", lw=3, legend=:bottomright);
    plot!(ts, dyn_with_ctr[2,:], label="Velocity", lw=3,
            legendfontsize=14, xticks=false, yticks=false);
    xlabel!("");
    title!("Observable Data", titlefontsize=24);

# Second: Controller
plot_control = plot(ts, vec(dyn_control_wall[2,:]), label="Ground",
                color=COLOR_DICT["control_true"], lw=3);
    plot_kick = plot!(ts, vec(dyn_control_kick[2,:]), label="Kick",
                color=COLOR_DICT["control_time"], lw=3,
                legendfontsize=14, xticks=false, yticks=false);
    xlabel!("Time", fontsize=16, legend=:bottomright);
    title!("Hidden Control Signals", titlefontsize=24);

# Creat the layout and plot
my_layout = @layout [p1; p2];
    p_final = plot(plot_data, plot_control, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_spatial_and_temporal_control.png";
# savefig(p_final, fname)
