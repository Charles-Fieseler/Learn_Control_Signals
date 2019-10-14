using PkgSRA
using Plots, Random, Distributions, Interpolations
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
include("../../utils/sindy_turing_utils.jl")
include("../../utils/sindy_statistics_utils.jl")
include("../../utils/main_algorithm_utils.jl")

#####
##### Define the controlled dynamical system
#####
# Load example problem
include(EXAMPLE_FOLDERNAME*"example_spiking_neuron.jl")
U_starts = 5.0
    amplitude = 40.0;
function U_func_current(t)
    ctr = zeros(2,1)
    ctr[1] = (t>U_starts ? amplitude : 0.0)
    return ctr
end

# Solve the system
dyn_with_ctr = solve_neuron_system(U_func_time=U_func_current)
plot(dyn_with_ctr)
dat = Array(dyn_with_ctr)

numerical_grad = numerical_derivative(dat, ts)
grad_true = dyn_with_ctr(ts, Val{1})

# Get just the control signals
U_true = hcat(U_func_current.(ts)...)


## Add a spiking control signal
num_ctr = 50;
    U_starts2 = rand(1, num_ctr) .* tspan[2]
    U_widths2 = [0.05];
    amplitudes = [200.0]
my_U_func_time2(t) = U_func_time_multivariate(t, zeros(2,1),
                            U_widths2, U_starts2,
                            F_dim=[1],
                            amplitudes=amplitudes)
U_func_total(t) = my_U_func_time2(t) + U_func_current(t)

dyn_with_spikes = solve_neuron_system(U_func_time=U_func_total)
plot(dyn_with_spikes)
dat2 = Array(dyn_with_spikes)

#####
##### First panels: Remove the control signal and fit a controlled model
#####
sindy_library = Dict("cross_terms"=>2, "constant"=>nothing);
chain_unctr, best_sindy_unctr = calc_distribution_of_models(
    dat, numerical_grad, sindy_library,
    val_list = calc_permutations(3,2)
)

plot(chain_unctr)
## Get the posterior distribution
(residual_unctr, sample_gradients, sample_noise, dat_grad) =
        calc_distribution_of_residuals(
        dat, numerical_grad, chain_unctr, 1:length(ts), best_sindy_unctr)
ctr_guess = process_residual(residual_unctr, sample_noise)

ind = 1:5000
i = 1
    plot(residual_unctr[i,ind], ribbon=500*sample_noise)
    plot!(U_true[:,ind]', label="True")
    title!("Residual and true control for channel $i")
i = 2
    plot(sample_gradients[i,ind], label="model", lw=2)
    plot!(numerical_grad[i,ind], label="data")
    title!("Data and model for channel $i")

## Subsample
# partial_accepted_ind = abs.(residual_unctr) .< (500*sample_noise)
# partial_accepted_ind = vec(prod(Int.(partial_accepted_ind), dims=1))
# _, _, all_starts, all_ends = calc_contiguous_blocks(
#           partial_accepted_ind,
#           minimum_length=1)

start_of_subsample = 1001
accepted_ind = subsample_using_residual(
            residual_unctr[:,start_of_subsample:end], sample_noise,
            noise_factor=500,
            min_length=10, shorten_length=2)
num_pts = 2000; start_ind = 101
subsample_ind = accepted_ind[start_ind:num_pts+start_ind-1] .+ start_of_subsample

U_sub = U_true[:,subsample_ind]
plot(U_sub')
    # plot!(residual[:,subsample_ind]', ribbon=sample_trajectory_noise)
    title!("Control signals in the subsampled dataset")
i = 2
    plot(subsample_ind,sample_gradients[i,subsample_ind], label="model", lw=2)
    plot!(subsample_ind,numerical_grad[i,subsample_ind], label="data")
    title!("Data and model for channel $i")

val_list = calc_permutations(5,2)
(final_sindy_model,best_criterion,all_criteria,_, _) =
     sindyc_ensemble(dat[:,subsample_ind],
                     numerical_grad[:, subsample_ind],
                     sindy_library, val_list,
                     selection_criterion=my_aicc,
                     sparsification_mode="num_terms",
                     selection_dist=Normal(0,10),
                     use_clustering_minimization=true)
print_equations(final_sindy_model)
scatter(sum.(val_list), all_criteria)


chain_ctr, best_sindy_ctr = calc_distribution_of_models(
    dat[:,subsample_ind],
    numerical_grad[:,subsample_ind],
    sindy_library,
    val_list = calc_permutations(4,2),
    chain_opt = (iterations=200, train_ind=1:num_pts)
)

print_equations(best_sindy_ctr)
# sindy_sample =  sindy_from_chain(best_sindy_ctr, chain_ctr)
# plot(chain_ctr)

# Calculate updated control signal
(residual_ctr, _, noise_ctr, _) =
        calc_distribution_of_residuals(
                dat, grad_true, chain_ctr,
                1:length(ts), best_sindy_ctr)
ctr_final = process_residual(residual_ctr, 100*noise_ctr)

plot(ctr_final', label="Learned", lw=3, color=:black)
    plot!(U_true', label="True")

# Get the actually controlled model
# val_list = calc_permutations(5,2)
# (final_sindy_model,best_criterion,all_criteria,_, _) =
#      sindyc_ensemble(dat, numerical_grad, sindy_library, val_list,
#                      U=ctr_final, ts=ts,
#                      selection_criterion=my_aicc,
#                      sparsification_mode="num_terms",
#                      selection_dist=Normal(0,1),
#                      use_clustering_minimization=false)

# scatter(sum.(val_list), all_criteria)
#
# prob = ODEProblem(final_sindy_model, dat[:,1], tspan)
# # ts2 = range(tspan[1], tspan[end], length=10000)
# dt = ts[2] - ts[1]
# sol = solve(prob, AB3(), dt=dt, saveat=ts)
# dat_final = Array(sol)
#
# plot(sol)




#####
##### Second panels: nonlinearity as well as control
#####
sindy_library = Dict("cross_terms"=>2, "constant"=>nothing);
chain_unctr2, best_sindy_unctr2 = calc_distribution_of_models(
    dat2, numerical_grad2, sindy_library,
    val_list = calc_permutations(3,2)
)

plot(chain_unctr)
## Get the posterior distribution
(residual_unctr, sample_gradients, sample_noise, dat_grad) =
        calc_distribution_of_residuals(
        dat, numerical_grad, chain_unctr, 1:length(ts), best_sindy_unctr)
ctr_guess = process_residual(residual_unctr, sample_noise)

# ind = 1:5000
# i = 1
#     plot(residual_unctr[i,ind], ribbon=500*sample_noise)
#     plot!(U_true[:,ind]', label="True")
#     title!("Residual and true control for channel $i")
# i = 2
#     plot(sample_gradients[i,ind], label="model", lw=2)
#     plot!(numerical_grad[i,ind], label="data")
#     title!("Data and model for channel $i")

start_of_subsample = 1001
accepted_ind = subsample_using_residual(
            residual_unctr[:,start_of_subsample:end], sample_noise,
            noise_factor=500,
            min_length=10, shorten_length=2)
num_pts = 2000; start_ind = 101
subsample_ind = accepted_ind[start_ind:num_pts+start_ind-1] .+ start_of_subsample

U_sub = U_true[:,subsample_ind]
plot(U_sub')
    # plot!(residual[:,subsample_ind]', ribbon=sample_trajectory_noise)
    title!("Control signals in the subsampled dataset")
i = 2
    plot(subsample_ind,sample_gradients[i,subsample_ind], label="model", lw=2)
    plot!(subsample_ind,numerical_grad[i,subsample_ind], label="data")
    title!("Data and model for channel $i")

val_list = calc_permutations(5,2)
(final_sindy_model,best_criterion,all_criteria,_, _) =
     sindyc_ensemble(dat[:,subsample_ind],
                     numerical_grad[:, subsample_ind],
                     sindy_library, val_list,
                     selection_criterion=my_aicc,
                     sparsification_mode="num_terms",
                     selection_dist=Normal(0,10),
                     use_clustering_minimization=true)
print_equations(final_sindy_model)
scatter(sum.(val_list), all_criteria)


chain_ctr, best_sindy_ctr = calc_distribution_of_models(
    dat[:,subsample_ind],
    numerical_grad[:,subsample_ind],
    sindy_library,
    val_list = calc_permutations(4,2),
    chain_opt = (iterations=200, train_ind=1:num_pts)
)

print_equations(best_sindy_ctr)
# sindy_sample =  sindy_from_chain(best_sindy_ctr, chain_ctr)
# plot(chain_ctr)

# Calculate updated control signal
(residual_ctr, _, noise_ctr, _) =
        calc_distribution_of_residuals(
                dat, grad_true, chain_ctr,
                1:length(ts), best_sindy_ctr)
ctr_final = process_residual(residual_ctr, 100*noise_ctr)

plot(ctr_final', label="Learned", lw=3, color=:black)
    plot!(U_true', label="True")



#####
##### Save
#####
this_dat_name = DAT_FOLDERNAME*"dat_neuron_"

# Raw data
fname = this_dat_name*"raw.bson";
@save fname dat true_grad numerical_grad dyn_with_ctr

# Naive model
# fname = this_dat_name*"naive_model.bson";
# @save fname dat_naive naive_model

# Controlled model
fname = this_dat_name*"controlled_model.bson";
@save fname chain_ctr ctr_final best_sindy_ctr
