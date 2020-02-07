using PkgSRA
using Plots, Random, Distributions, Interpolations
pyplot()
Random.seed!(11)
using BSON: @save
include("../scripts/paper_settings.jl");
# include("../../utils/sindy_turing_utils.jl")
# include("../../utils/sindy_statistics_utils.jl")
# include("../../utils/main_algorithm_utils.jl")


################################################################################
#####
##### SIR example
#####
include(EXAMPLE_FOLDERNAME*"example_sir.jl")

# Define the multivariate forcing function
num_ctr = 3;
    U_starts = rand(1, num_ctr) .* tspan[2]/2
    U_widths = 0.6;
    amplitude = 100.0
my_U_func_time2(t) = U_func_time(t, u0,
                        U_widths, U_starts,
                        F_dim=1,
                        amplitude=amplitude)

# Get data
sol = solve_sir_system(U_func_time=my_U_func_time2)
dat = Array(sol)
plot(ts,dat[1,:], label="S");
    plot!(ts,dat[2,:], label="I");
    plot!(ts,dat[3,:], label="R")
numerical_grad = numerical_derivative(dat, ts)
true_grad = core_dyn_true(dat)
plot(ts, true_grad[2,:]);plot!(ts, numerical_grad[2,:])

## Also get baseline true/ideal cases
# Uncontrolled
# dat_raw = Array(solve_sir_system())
# numerical_grad_raw = numerical_derivative(dat_raw, ts)
# true_grad_raw = zeros(size(dat))
# for i in 1:size(dat,2)
#     true_grad_raw[:,i] = sir_system(dat_raw[:,i], Float64.(p), [0])
# end

# True control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time2(t)
end
plot(ts,dat[1,:], label="S");
    plot!(ts,dat[2,:], label="I");
    plot!(ts,dat[3,:], label="R")
    plot!(ts,U_true', label="Control")


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
                    selection_dist=Normal(0.0,100),
                    use_clustering_minimization=true)
println("Naive SINDy model:")
    print_equations(sindy_unctr, digits=5)
println("True equations are:")
    print_equations(core_dyn_true, digits=5)
scatter(sum.(val_list), all_criteria)
    title!("AIC for various sparsities (naive model)")
    xlabel!("Number of nonzero terms")


# Integrate this (poor) model
# Callback aborts if it blows up
condition(u,t,integrator) = any(abs.(u).>1e4)
cb = DiscreteCallback(condition, terminate!)
prob_unctr = ODEProblem(sindy_unctr, u0, tspan, [0], callback=cb)
sindy_dat_unctr = Array(solve(prob_unctr, Tsit5(), saveat=ts));

# let d = sindy_dat_unctr, d2 = dat
#     i = 1;
#     plot(d[i,:], label="Naive model");#plot!(d[2,:]);plot!(d[3,:])
#     plot!(d2[i,:], label="Data");#plot!(d2[2,:]);plot!(d2[3,:])
# end


###
### Calculate the residuals
###
# NOTE: do not use Turing, for speed
sindy_grad1 = sindy_unctr(dat, 0)
residual = numerical_grad .- sindy_grad1
noise_guess = 2*abs(median(residual[1,:])) # Really should use Turing here

plot(residual[3,:], label="Residual")
    hline!([noise_guess], label="Noise line");
    hline!([-noise_guess], label="Noise line")
    # title!("Residual")

ctr_guess = process_residual(residual, noise_guess)


###
### Subsample based on residual
###
accepted_ind = subsample_using_residual(residual,
            noise_guess, min_length=4)

num_pts = 2000
start_ind = 1
subsample_ind = accepted_ind[start_ind:num_pts+start_ind-1]
dat_sub = dat[:,subsample_ind]
grad_sub = numerical_grad[:,subsample_ind]

# Plot the subsampled points
i = 2
    plot(dat[i,:])
    scatter!(subsample_ind, dat_sub[i,:], color=:blue)
    plot!(U_true[i,:])
    title!("Subsampled points for index $i")

# Any control signal in the subset?
# U_sub = U_true[:,subsample_ind]
# plot(U_sub')
#     title!("Control signals in the subsampled dataset")

# val_list = calc_permutations(6,3)
# val_list = combinations(1:5, 3)
val_list = Iterators.product(1:3,1:3,1:3) # DEBUG
(sindy_sub,best_criterion,all_criteria,all_models,best_index) =
    sindyc_ensemble(dat_sub, grad_sub, sindy_library, val_list,
                    selection_criterion=my_aicc,
                    sparsification_mode="num_terms",
                    selection_dist=Normal(0.0,2*noise_guess),
                    use_clustering_minimization=true)
println("Best index is $best_index:")
    print_equations(sindy_sub, digits=5)
println("True equations are:")
    print_equations(core_dyn_true, digits=5)
scatter(sum.(val_list), all_criteria)
    title!("AIC for various sparsities")
    xlabel!("Number of nonzero terms")



# Plot a function for the subsampled points
# f(S, I, R) = -0.00027*I.*I .- 0.0002*I.*R
# g(S, I, R) = -0.2*I
# i = 2
#     plot(g(dat[1,:], dat[2,:], dat[3,:]), label="True")
#     scatter!(subsample_ind, g(dat_sub[1,:], dat_sub[2,:], dat_sub[3,:]))
#     plot!(f(dat[1,:], dat[2,:], dat[3,:]), label="Found")
#     title!("Subsampled points for index $i")
