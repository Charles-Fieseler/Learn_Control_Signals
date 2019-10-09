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
# Load helper functions
include("../../utils/sindy_statistics_utils.jl")
include("../../utils/sindy_turing_utils.jl")
include("../scripts/paper_settings.jl")
# Load example problem
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

# Define the multivariate forcing function
num_ctr = 50;
    U_starts = rand(3, num_ctr) .* tspan[2]
    U_widths = [0.1, 0.1, 0.0];
    amplitudes = [150.0, 100.0, 0.0]
my_U_func_time2(t, u) = U_func_time_multivariate(t, u,
                        U_widths, U_starts,
                        F_dim=[1, 2, 3],
                        amplitudes=amplitudes)

#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time2)
dat = Array(sol)
# plot(dat[1,:],dat[2,:],dat[3,:],label="Data")
# Derivatives
numerical_grad = numerical_derivative(dat, ts)

# ## Also get baseline true/ideal cases
# # Uncontrolled
# dat_raw = Array(solve_lorenz_system())
# numerical_grad_raw = numerical_derivative(dat_raw, ts)
# true_grad_raw = zeros(size(dat))
# for i in 1:size(dat,2)
#     true_grad_raw[:,i] = lorenz_system(dat_raw[:,i], Float64.(p), [0])
# end
# # dat_raw += 0.1*randn(size(dat_raw))
#
# val_list = calc_permutations(5,3)
# (best_model_raw,best_criterion,all_criteria,all_models) =
#     sindyc_ensemble(dat_raw, true_grad_raw, sindy_library, val_list,
#                     selection_criterion=my_aicc,
#                     sparsification_mode="num_terms",
#                     selection_dist=Normal(0.0,20))
# print_equations(best_model_raw)
# scatter(sum.(val_list), all_criteria)

# Controlled; true variables
true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end
let d = dat
    plot(d[1,:],d[2,:],d[3,:],label="Data")
end

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t, dat[:,i])
end
plot(U_true')

#####
##### Calculate distribution of residuals
#####
# Turing.setadbackend(:forward_diff)
# CHEATING: known structure
#   Calculate posterior distribution of parameters
# chain, train_ind = generate_chain(dat,
#                         numerical_grad, lorenz_grad_residual)
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);
Random.seed!(13)

# Upgrade: Use AIC
val_list = calc_permutations(5,3)
(sindy_unctr,best_criterion,all_criteria,all_models) =
    sindyc_ensemble(dat, numerical_grad, sindy_library, val_list,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms",
                    selection_dist=Normal(0.0,20),
                    use_clustering_minimization=true)
print_equations(sindy_unctr)
# scatter(sum.(val_list), all_criteria)

# Generate a full trajectory from SINDy model
condition(u,t,integrator) = any(abs.(u).>2e2)
cb = DiscreteCallback(condition, terminate!)
prob_unctr = ODEProblem(sindy_unctr, u0, tspan, [0], callback=cb)
dat_unctrL2 = Array(solve(prob_unctr, Tsit5(), saveat=ts));

plot(dat[1,:],dat[2,:],dat[3,:],label="Data")
    plot!(dat_unctrL2[1,:],dat_unctrL2[2,:],dat_unctrL2[3,:],label="Turing model with enforced zeros")
    title!("Integrated SINDy model")

# grad_unctr = sindy_unctr(dat)
# let d = grad_unctr, d0 = numerical_grad
#     plot(d[1,:],d[2,:],d[3,:],label="Model")
#     plot!(d0[1,:],d0[2,:],d0[3,:],label="Data")
#     title!("Gradients (L2 SINDy model)")
# end
turing_unctr = convert_sindy_to_turing_enforce_zeros(sindy_unctr;
                                dat_noise_prior=Normal(0.0, 20.0),
                                coef_noise_std=2.0)
chain = generate_chain(dat, numerical_grad, turing_unctr,
                            iterations=200,
                            num_training_pts=100, start_ind=101)[1]
turing_unctr_sample = sindy_from_chain(sindy_unctr, chain,
                                        enforced_zeros=true)
# plot(chain["all_coef[3]"])
# Generate a single full sample trajectory (FROM TURING)
# prob_unctr = ODEProblem(sindy_unctr, u0, tspan, [0], callback=cb)
prob_unctr = ODEProblem(turing_unctr_sample, u0, tspan, [0], callback=cb)
sol_unctr = solve(prob_unctr, Tsit5(), saveat=ts);
dat_unctr = Array(sol_unctr)

plot(dat[1,:],dat[2,:],dat[3,:],label="Data")
    plot!(dat_unctr[1,:],dat_unctr[2,:],dat_unctr[3,:],label="Turing model with enforced zeros")
    title!("Integrated Turing model")

# Get the full posterior distribution
sample_ind = 1:length(ts)
sample_gradients3d, sample_noise = sample_sindy_posterior_grad(chain,
            dat, sample_ind, sindy_unctr)
# let y = sample_gradients3d[1,:,:], y2 = sample_gradients3d[2,:,:], n=numerical_grad
#     plot3d(y[:,1], y[:,2], y[:,3], label="Naive Sample 1")
#     plot3d!(y2[:,1], y2[:,2], y2[:,3], label="Naive Sample 2")
#     plot3d!(n[1,:], n[2,:], n[3,:], label="Data")
#     title!("Sampled Turing Gradients")
# end

# Calculate the residuals
sample_gradients = transpose(drop_all_1dims(mean(sample_gradients3d, dims=1)))
    sample_gradients_std = transpose(drop_all_1dims(std(sample_gradients3d, dims=1)))
    sample_trajectory_noise = mean(sample_noise)
dat_grad = numerical_grad[:, sample_ind]

residual = dat_grad .- sample_gradients
ctr_guess = process_residual(residual, sample_trajectory_noise)

# Visualization
ind = 101:1000
# ind = sample_ind
i = 2
    plot(dat_grad[i,ind], label="Data gradient")
    plot!(sample_gradients[i,ind], label="Sample gradient")
i = 2
    plot(residual[i,ind], ribbon=sample_trajectory_noise)
    plot!(U_true[i,ind], label="True")
    title!("Residual and true control")
i = 1
    plot(sample_gradients_std[i,ind])
    plot!(U_true[i,ind], label="True")
    title!("Residual and true control")
i = 2
    plot(residual[i,ind],ribbon=
            sample_trajectory_noise.+sample_gradients_std[i,ind])
    plot!(U_true[i,ind], label="True")
    title!("Residual and true control (with std)")
# i = 1
#     plot(ctr_guess[i,ind], label="Control guess")
#     plot!(U_true[i,ind], label="True", show=true)




#####
##### INSTEAD: subsample the data based on the above residual
#####

# Get "confident" indices
accepted_ind = subsample_using_residual(residual,
            sample_trajectory_noise, min_length=4)

num_pts = 300
subsample_ind = accepted_ind[1:num_pts]
dat_sub = dat[:,subsample_ind]
grad_sub = numerical_grad[:,subsample_ind]

# Any control signal in the subset?
U_sub = U_true[:,subsample_ind]
plot(U_sub')
    title!("Control signals in the subsampled dataset")

# SINDY SETUP
val_list = calc_permutations(5,3)
(sindy_sub,best_criterion,all_criteria,all_models) =
    sindyc_ensemble(dat_sub, grad_sub, sindy_library, val_list,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms",
                    selection_dist=Normal(0.0,sample_trajectory_noise),
                    use_clustering_minimization=true)
print_equations(sindy_sub)
scatter(sum.(val_list), all_criteria)
    title!("AIC for various sparsities")
    xlabel!("Number of nonzero terms")

# Generate a single trajectory
prob_ctr = ODEProblem(sindy_sub, u0, tspan, [0], callback=cb)
dat_ctrL2 = Array(solve(prob_ctr, Tsit5(), saveat=ts));

plot(dat[1,:],dat[2,:],dat[3,:],label="Data")
    plot!(dat_ctrL2[1,:],dat_ctrL2[2,:],dat_ctrL2[3,:],label="Turing model with enforced zeros")
    title!("Integrated Turing model (after control)")
# TURING ANALYSIS
turing_sub = convert_sindy_to_turing_enforce_zeros(sindy_sub;
                                dat_noise_prior=Normal(0.0, 5.0),
                                coef_noise_std=1.0)
chain_sub = generate_chain(dat, numerical_grad, turing_sub,
                            train_ind=subsample_ind,
                            iterations=200)[1]
turing_sub_sample = sindy_from_chain(sindy_sub, chain_sub,
                                        enforced_zeros=true)
# Generate test trajectories from the CORRECTED posterior
sample_gradients3d_ctr, sample_noise_ctr =
            sample_sindy_posterior_grad(chain_ctr,
                    dat, sample_ind, sindy_ctr)
# Calculate the residuals
sample_trajectory_noise_ctr = mean(sample_noise_ctr)
# Generate a single full sample trajectory (FROM TURING)
prob_ctr = ODEProblem(turing_sub_sample, u0, tspan, [0], callback=cb)
dat_ctr = Array(solve(prob_ctr, Tsit5(), saveat=ts));


println("SINDy of subset")
print_equations(sindy_sub)
# println("Turing-SINDy of subset")
# print_equations(turing_sub_sample)
println("True")
print_equations(core_dyn_true)




#####
##### Create a new model, subtracting the residual
#####
# Try to predict the MODIFIED GRADIENT from data
# y = numerical_grad[:,sample_ind] .- ctr_guess
# sindy_ctr =  sindyc(dat, y, library=sindy_library, use_lasso=true)
# turing_ctr = convert_sindy_to_turing_enforce_zeros(sindy_ctr;
#                                 dat_noise_prior=Normal(0.0, 1.0),
#                                 coef_noise_std=0.1)
# chain_ctr = generate_chain(dat, y, turing_ctr,
#                             iterations=200,
#                             num_training_pts=200, start_ind=101)[1]
# turing_ctr_sample = sindy_from_chain(sindy_ctr, chain_ctr,
#                                         enforced_zeros=true)
#
# # Generate test trajectories from the CORRECTED posterior
# sample_gradients3d_ctr, sample_noise_ctr =
#             sample_sindy_posterior_grad(chain_ctr,
#                     dat, sample_ind, sindy_ctr)
# # Calculate the residuals
# sample_trajectory_noise_ctr = mean(sample_noise_ctr)
# # Generate a single full sample trajectory (FROM TURING)
# prob_ctr = ODEProblem(turing_ctr_sample, u0, tspan, [0], callback=cb)
# sol_ctr = solve(prob_ctr, Tsit5(), saveat=ts);
# dat_ctr = Array(sol_ctr)
#
# plot(dat[1,:],dat[2,:],dat[3,:],label="Data")
#     plot!(dat_ctr[1,:],dat_ctr[2,:],dat_ctr[3,:],label="Turing model with enforced zeros")
#     title!("Integrated Turing model (with control)")
# let y = sample_gradients3d_ctr[1,:,:]', y2 = sample_gradients3d_ctr[2,:,:]', n=numerical_grad
#     plot3d(y[1,:], y[2,:], y[3,:], label="Corrected Sample 1")
#     plot3d!(y2[1,:], y2[2,:], y2[3,:], label="Corrected Sample 2")
#     plot3d!(n[1,:], n[2,:], n[3,:], label="Data")
# end


#####
##### Save everything; plotted in a different script
#####
this_dat_name = DAT_FOLDERNAME*"TMP_dat_flowchart_inkscape_"

# Intermediate data
fname = this_dat_name*"intermediate.bson";
@save fname chain chain_sub sol sample_gradients sample_trajectories_ctr sample_noise sample_noise_ctr sample_ind numerical_grad

# Original ODE and controller variables
fname = this_dat_name*"ode_vars.bson";
@save fname dat dat_grad true_grad U_true

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars.bson";
@save fname dat_unctrL2 dat_unctr sample_trajectory_noise residual accepted_ind

# Bayesian variables 2: after control
fname = this_dat_name*"ctr_vars.bson";
@save fname sindy_sub dat_ctrL2 dat_ctr sample_trajectory_noise_ctr #sample_trajectories_ctr

# TODO: NN variables
