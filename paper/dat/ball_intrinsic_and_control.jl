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
include(EXAMPLE_FOLDERNAME*"example_falling_ball.jl")
U_starts = [6.0]
    U_widths = 0.2;
    amplitude = 75.0;
U_func_kick(t, u) = U_func_time(t, u,
                                U_widths, U_starts,
                                F_dim=2,
                                amplitude=amplitude)

# Define the time dependent forcing function

U_func_wall(X) = U_func_spring(X, k=1e3, r=1.0);

# Solve the system
dyn_with_ctr = solve_ball_system(U_func_space=U_func_wall,
                                 U_func_time=U_func_kick)
dat = Array(dyn_with_ctr)
dat_with_const = vcat(dat, ones(1,size(dat,2)))
numerical_grad = numerical_derivative(dat, ts)

# plot(dat')

# Get just the control signals
dyn_control_wall = zeros(size(dat))
for i in 1:size(dat, 2)
    dyn_control_wall[:,i] = U_func_wall(dat[:, i])
end

dyn_control_kick = zeros(size(dat))
for (i, t) in enumerate(ts)
    dyn_control_kick[:,i] = U_func_kick(t, dat[:, i])
end

#####
##### Fit a naive linear model (with constant)
#####
naive_model = dmdc(dat_with_const, numerical_grad)[1]

f(u,p,t) = vcat(naive_model * u, [1])
prob = ODEProblem(f, vcat(u1,[1]), tspan, p1)
dat_naive = Array(solve(prob, Tsit5(), saveat=ts))

# plot(dat_naive[1,:]); plot!(dat[1,:])

#####
##### Remove the control signal and fit a controlled model
#####

## NOTE: I'm using SINDy with degree 1 cross terms... i.e. a linear model!
sindy_library = Dict("constant"=>nothing);
chain_unctr, best_sindy_unctr = calc_distribution_of_models(
    dat, numerical_grad, sindy_library,
    val_list = calc_permutations(3,2)
)

# plot(chain_unctr)
## Get the posterior distribution
(residual_unctr, sample_gradients, sample_noise, dat_grad) =
        calc_distribution_of_residuals(
        dat, numerical_grad, chain_unctr, 1:length(ts), best_sindy_unctr)
ctr_guess = process_residual(residual_unctr, sample_noise)

# U_true = vcat(dyn_control_kick, dyn_control_wall)
# ind = 1:500
# i = 2
#     plot(residual_unctr[i,ind], ribbon=sample_noise)
#     plot!(U_true[:,ind]', label="True")
#     title!("Residual and true control")

## Subsample
accepted_ind = subsample_using_residual(
            residual_unctr, sample_noise,
            min_length=1, shorten_length=2)
num_pts = 400; start_ind = 101
subsample_ind = accepted_ind[start_ind:num_pts+start_ind-1]

# U_sub = U_true[:,subsample_ind]
# plot(U_sub')
#     # plot!(residual[:,subsample_ind]', ribbon=sample_trajectory_noise)
#     title!("Control signals in the subsampled dataset")

chain_ctr, best_sindy_ctr = calc_distribution_of_models(
    dat[:,subsample_ind],
    numerical_grad[:,subsample_ind],
    sindy_library,
    val_list = calc_permutations(3,2),
    chain_opt = (iterations=200, train_ind=1:num_pts)
)

# print_equations(best_sindy_ctr)
# plot(chain_ctr)

# Calculate updated control signal
(residual_ctr, ctr_final, _, _, _) =
        calc_distribution_of_residuals(
        dat, numerical_grad, chain_ctr, 1:length(ts), best_sindy_ctr)

plot(ctr_final', label="Learned", lw=3, color=:black)
    plot!(U_true', label="True")

# Get the actually controlled model
val_list = calc_permutations(3,2)
(final_sindy_model,best_criterion,all_criteria,_, _) =
     sindyc_ensemble(dat, numerical_grad, sindy_library, val_list,
                     U=ctr_final, ts=ts,
                     selection_criterion=my_aicc,
                     sparsification_mode="num_terms",
                     selection_dist=Normal(0,1),
                     use_clustering_minimization=false)

scatter(sum.(val_list), all_criteria)
F = generate_map_rows_function(
        (d)->LinearInterpolation(ts,d), ctr_final)
# plot(vcat.(F.(ts)...))
# F(0)
f(u,p,t) = final_sindy_model(u) .+ F(t)
prob = ODEProblem(final_sindy_model, dat[:,1], tspan)
# ts2 = range(tspan[1], tspan[end], length=10000)
sol = solve(prob, Tsit5(), saveat=ts)
dat_final = Array(sol)

plot(sol)
#####
##### Save
#####
this_dat_name = DAT_FOLDERNAME*"dat_ball_"

# Raw data
fname = this_dat_name*"raw.bson";
@save fname dat numerical_grad dyn_control_kick dyn_control_wall

# Naive model
fname = this_dat_name*"naive_model.bson";
@save fname dat_naive naive_model

# Controlled model
fname = this_dat_name*"controlled_model.bson";
@save fname chain_ctr ctr_final
