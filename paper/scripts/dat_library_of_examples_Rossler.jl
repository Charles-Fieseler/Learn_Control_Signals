using PkgSRA
using Plots, Random, Distributions, Interpolations
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
include("../../utils/sindy_turing_utils.jl")
# include("../../utils/sindy_statistics_utils.jl")
# include("../../utils/main_algorithm_utils.jl")


################################################################################
#####
##### Rossler example
#####
include(EXAMPLE_FOLDERNAME*"example_rossler.jl")

# Define the multivariate forcing function
num_ctr = 10;
    U_starts = rand(3, num_ctr) .* tspan[2]
    U_widths = [0.0, 0.0, 0.5];
    amplitudes = [1.0, 1.0, 1000.0]
my_U_func_time2(t) = U_func_time_multivariate(t, u0,
                        U_widths, U_starts,
                        F_dim=[1,2,3],
                        amplitudes=amplitudes)

# Get data
sol = solve_rossler_system(U_func_time=my_U_func_time2)
dat = Array(sol)
plot3d(dat[1,:],dat[2,:],dat[3,:],label="Data")
# plot(ts,dat[1,:]);plot!(ts,dat[2,:]);plot!(ts,dat[3,:])
# plot(ts,dat[3,:])
numerical_grad = numerical_derivative(dat, ts)

## Also get baseline true/ideal cases
# Uncontrolled
dat_raw = Array(solve_rossler_system())
numerical_grad_raw = numerical_derivative(dat_raw, ts)
true_grad_raw = zeros(size(dat))
# for i in 1:size(dat,2)
#     true_grad_raw[:,i] = rossler_system(dat_raw[:,i], Float64.(p), [0])
# end

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t)
end
plot(ts,dat[1,:]);plot!(ts,dat[2,:]);plot!(ts,dat[3,:])
    plot!(ts,U_true')



###
### Fit a naive model
###
sindy_library = Dict("cross_terms"=>2,"constant"=>nothing);

# Upgrade: Use AIC
# val_list = calc_permutations(5,3)
# val_list = combinations(1:5, 3)
val_list = Iterators.product(1:3,1:3,1:3) # DEBUG
(sindy_unctr,best_criterion,all_criteria,all_models) =
    sindyc_ensemble(dat, numerical_grad, sindy_library, val_list,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms",
                    selection_dist=Normal(0.0,20),
                    use_clustering_minimization=true)
print_equations(sindy_unctr)

# Integrate this (poor) model
# Callback aborts if it blows up
condition(u,t,integrator) = any(abs.(u).>1e4)
cb = DiscreteCallback(condition, terminate!)
prob_unctr = ODEProblem(sindy_unctr, u0, tspan, [0], callback=cb)
sindy_dat_unctr = Array(solve(prob_unctr, Tsit5(), saveat=ts));

let d = sindy_dat_unctr, d2 = dat
    plot(d[1,:]);plot!(d[2,:]);plot!(d[3,:])
    plot!(d2[1,:]);plot!(d2[2,:]);plot!(d2[3,:])
end
dat_unctr = sindy_dat_unctr


###
### Calculate the residuals
###
# NOTE: do not use Turing, for speed
sindy_grad1 = sindy_unctr(dat, 0)
residual = numerical_grad .- sindy_grad1
noise_guess = 10*abs(median(residual[3,:]))

scatter(residual[3,:])
    hline!([-noise_guess noise_guess])

ctr_guess = process_residual(residual, noise_guess)


###
### Subsample based on residual
###
accepted_ind = subsample_using_residual(residual,
            noise_guess, min_length=4, shorten_length=0)

num_pts = 1000
start_ind = 1
subsample_ind = accepted_ind[start_ind:num_pts+start_ind-1]
dat_sub = dat[:,subsample_ind]
grad_sub = numerical_grad[:,subsample_ind]

# Plot the subsampled points
i = 3;
    plot(dat[i,:], label="Data; NOT residual")
    scatter!(subsample_ind, dat_sub[i,:], color=:blue, label="Subsampled points")
    plot!(U_true[i,:], label="Control Signal")

# Better SINDy model
# val_list = calc_permutations(7,3)
# val_list = combinations(1:5, 3)
val_list = Iterators.product(1:3,1:3,1:3) # DEBUG
(sindy_sub,best_criterion,all_criteria,all_models,best_index) =
    sindyc_ensemble(dat_sub, grad_sub, sindy_library, val_list,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms",
                    selection_dist=Normal(0.0,0.1*noise_guess),
                    use_clustering_minimization=false)
println("Best index is $best_index")
print_equations(sindy_sub)
print("True equations")
print_equations(core_dyn_true)
scatter(sum.(val_list), all_criteria)
    title!("AIC for various sparsities")
    xlabel!("Number of nonzero terms")


# Generate the attractor using this system
condition(u,t,integrator) = any(abs.(u).>1e3)
cb = DiscreteCallback(condition, terminate!)
prob_ctr = ODEProblem(sindy_sub, u0, tspan, [0], callback=cb)
sindy_dat_ctr = solve(prob_ctr, Tsit5(), saveat=ts);
dat_ctr = Array(sindy_dat_ctr)

plot3d(sindy_dat_ctr[1,:], sindy_dat_ctr[2,:], sindy_dat_ctr[3,:])

# Also generate the derivatives
sindy_grad_ctr = sindy_sub(dat, 0)

# And then the guessed control signal

ctr_guess2 = process_residual(numerical_grad.-sindy_grad_ctr, noise_guess)


#####
##### Save Rossler data
#####
this_dat_name = DAT_FOLDERNAME*"TMP_dat_library_of_examples_rossler"

# Uncontrolled data
fname = this_dat_name*"uncontrolled.bson"
@save fname dat_raw numerical_grad_raw

# Original ODE and controller variables
fname = this_dat_name*"ode_vars.bson";
@save fname dat U_true #dat_grad

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars_bayes.bson";
@save fname dat_unctr noise_guess residual accepted_ind ctr_guess

# SINDY variables 2: after control
fname = this_dat_name*"ctr_vars_sindy.bson";
@save fname dat_ctr sindy_grad_ctr ctr_guess2

# Bayesian variables 2: after control
# fname = this_dat_name*"ctr_vars_turing.bson";
# # @save fname dat_ctr sample_trajectory_noise_ctr sample_gradients_ctr turing_dat_ctr ctr_guess2
# @save fname dat_ctr
