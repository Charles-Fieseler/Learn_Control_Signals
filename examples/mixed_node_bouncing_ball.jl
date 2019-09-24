using PkgSRA
using Plots, Random
using Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
using DSP
pyplot()
Random.seed!(11)

#####
##### Generate true dynamics matrix and data
#####
# Note: continuous version

# Generate the control signal starts
# include("example_forcing_functions.jl")
include("example_falling_ball.jl")
u0 = u1;
# dat_no_wall = solve_ball_system()
# plot(dat_no_wall[1,:], label="Ball position")


# dat = solve_deformed_ball_system()

U_func(X) = U_func_spring(X);
dat = solve_ball_system(U_func_time_trivial,
                        U_func)
# dat = Array(sol)

# Add a row of constants to the data (not actually needed here)
n, m = size(dat)
dat_and_const = vcat(dat, ones(1,m))

#####
##### Mini-SRA: initial guess for dynamics vs. control
#####
dat_grad = numerical_derivative(dat, ts)
# Plot this numerical derivative vs. the truth
plot(dat_grad[2,:], label="Numerical derivative")
# plot!(core_dyn_true(dat)[2,:], label="True derivative")

U_guess = sra_dmd(dat_and_const, dat_grad,
                    quantile_threshold=0.8)[1]
A, B = dmdc(dat_and_const, dat_grad, U_guess)
# Loop a couple of times
for i in 1:20
    pt = plot(U_guess[2,:], label="Residual")
    # plot!(U_func.(ts), label="True")
    title!("Iteration $i")
    display(pt)
    # Update guess for control signal
    global U_guess = calculate_residual(dat_and_const,
                                        dat_grad,
                                        A)
    sparsify_signal!(U_guess, quantile_threshold=0.8)
    # Update guess for dynamics
    global A, B = dmdc(dat_and_const, dat_grad, U_guess)
end

# Filter the signal to be easier for the NN to learn
# myfilter = digitalfilter(Lowpass(0.5),Butterworth(2));
# U_filter = zeros(size(U_guess))
# for i in 1:size(U_guess, 1)
#     U_filter[i,:] = filtfilt(myfilter, U_guess[i,:]);
# end
# U_filter *= maximum(abs.(U_guess))/maximum(abs.(U_filter))
U_true = zeros(size(dat))
for i in 1:size(dat,2)
    U_true[:,i] = U_func(dat[:,i])
end
# U_true = [U_func(d) for d in dat]
plot(U_true[2,:], label="Control Signal")
plot!(U_guess[2,:], label="Raw Guessed Control Signal")
# plot!(U_filter[2,:], label="Filtered Guessed Control Signal")
U_filter = U_guess


#####
##### Learn a NN that can reconstruct the SRA control signal
#####
# Now, a function of phase space
nn_dim = 24
U_dim = 1 # Expect that the true dimensionality is 1
# Note: need to initialize the first layer so that the whole space is used
ctr_dyn = Chain(Dense(n, nn_dim),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, U_dim),
                Dense(U_dim,2)) # Equivalent to the B matrix
ps = Flux.params(ctr_dyn)
loss_U_filter() = sum(abs2,ctr_dyn(dat) .- U_filter)

loss_to_use = loss_U_filter

cb_ctr = function (i=2)
    pt = plot(Flux.data(ctr_dyn(dat)[i,:]), label="Prediction")
    plot!(U_filter[i,:], label="Guess for control signals")
    display(pt)
    return pt
end
cb_ctr(2)

# Initial fast learning
tol = 200
dat_dummy = Iterators.repeated((), 100)
opt = ADAM(0.1)
while loss_to_use() > 5*tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    cb_ctr(2)
end
opt = ADAM(0.01)
while loss_to_use() > tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
    cb_ctr(2)
end
println("Finished learning control signal")

#####
##### As a sanity check, do the NN syntax with the true dynamics
#####
# B_true = Float32.([0.; 1.])
# function dudt_true(u::AbstractArray,p,t)
#     core_dyn_true(u) + B_true*Float32(U_func(t))
# end
# p = param(Float32[0.0]) # Unused
# prob_true = ODEProblem(dudt_true,u0,tspan,p)
# sol = diffeq_rd(p,prob_true,Tsit5(), saveat=ts)
# plot(Flux.data.(Array(sol))[2,:], label="Ideal Simulation")
# plot!(dat[2,:], label="Data")
# title!("True dynamics")
#
# best_loss = sum(abs2,Flux.data.(Array(sol)) .- dat)
# println("The best loss achievable is $best_loss")

#####
##### DEFINE the unified mixed NODE
#####
# Part 1: Linear dynamics; don't predict time
# Note: use actual constructor, not the convenience one
core_dyn = Dense(param(Float32.(A[:, 1:n])),
                 param(Float32.(A[:, end])),
                 identity)
# Part 2: Neural network for control signal; initialized above
p = param(Float32[1.0; 1.0])

function dudt_(u::AbstractArray,p,t)
    core_dyn(u) + p.*ctr_dyn(u)
end

prob = ODEProblem(dudt_,u0,tspan,p)
# Check the initial solution
# sol = diffeq_rd(p,prob,Tsit5(), saveat=ts)
# plot(Flux.data.(Array(sol))[2,:], label="Simulation")
# plot!(dat[2,:], label="Data")

_x = param(u0) # The initial condition can be learned too

function predict_rd()
    Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=_x, saveat=ts))
end
function loss_rd(λ1=0.0001, λ2=0.001)
    return sum(abs2,predict_rd() .- dat) +      # Good fit
            λ1*loss_U_filter() +                # Control signal close to guess
            λ2*sum(abs,ctr_dyn(dat))   # L1 norm on ctr
end

cb_plot = function (i=1)
  # display(loss_rd())
  pt = plot(ts, Flux.data(predict_rd()[i,:]), label="Prediction")
  plot!(ts, dat[i,:], label="Data")
  display(pt)
  return pt
end

#####
##### SOLVE the mixed NODE
#####
ps = params(core_dyn,ctr_dyn,p);

# Display the ODE with the current parameter values.
cb_plot(1)
# Initial quick learning
opt = ADAM(1e-6)
dat_dummy = Iterators.repeated((), 10)
Flux.train!(loss_rd, ps,
            dat_dummy, opt, cb = ()->@show loss_rd())
cb_plot(1)
# Run for a long time
opt = ADAM(1e-9)
tol = 50
dat_dummy = Iterators.repeated((), 5)
while loss_rd() > tol
    Flux.train!(loss_rd, ps,
                dat_dummy, opt, cb = ()->@show loss_rd())
    cb_plot(1)
    break
end


#####
##### Instead: SOLVE the mixed NODE via alternating
#####

# Pure loss
loss_dat() = sum(abs2,predict_rd() .- dat)
loss_ctr(λ=0.1) = loss_dat() + λ*sum(abs,ctr_dyn(dat))

function plot_dat_and_ctr()
    pt1 = cb_plot();
    pt2 = cb_ctr();
    pt = plot(pt1, pt2, layout=@layout [a;b]);
    display(pt)
end

ps_core = params(core_dyn);
ps_ctr = params(ctr_dyn,p);

plot_dat_and_ctr()
dat_dummy = Iterators.repeated((), 10)
tol = 150;
opt = ADAM(1e-10)
while loss_dat() > tol
    println("Learning core dynamics; direct L2 loss")
    Flux.train!(loss_dat, ps_core,
                dat_dummy, opt, cb = ()->@show loss_dat())
    plot_dat_and_ctr()

    println("Learning Control signal; direct L2 loss with L1 penalty")
    Flux.train!(loss_ctr, ps_ctr,
                dat_dummy, opt, cb = ()->@show loss_dat())
    plot_dat_and_ctr()

    # break
end
