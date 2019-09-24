using PkgSRA
using Plots, Random
using Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include(EXAMPLE_FOLDERNAME*"example_mass_spring.jl")

# Define the time dependent forcing function
# num_ctr = 2;
# U_starts = rand(num_ctr,1) .* tspan[2]
# U_width = 0.05;
# U_func(t) = U_func_time(t, U_width, U_starts)
U_func = spring_forcing_example(tspan)


# Solve the system
dat = solve_msd_system(U_func)
# Add a constant to the data
n, m = size(dat)
dat_and_const = vcat(dat, ones(1,m))

#####
##### SRA: initialize the separation for dynamics vs. control
#####
dat_grad = numerical_derivative(dat, ts)

U_guess = sra_dmd(dat_and_const, dat_grad, quantile_threshold=0.8)[1]
A, B = dmdc(dat_and_const, dat_grad, U_guess)
# Loop a couple of times
for i in 1:20
    # Update guess for control signal
    global U_guess = calculate_residual(dat_and_const,
                                        dat_grad,
                                        A)
    sparsify_signal!(U_guess, quantile_threshold=0.8)
    # Update guess for dynamics
    global A, B = dmdc(dat_and_const, dat_grad, U_guess)
end

#####
##### DATA: guess for intrinsic dynamics
####
function dudt_(u::AbstractArray,p,t)
    A[:, 1:2]*u + A[:, end]
end
u0 = dat[:,1]
prob = ODEProblem(dudt_,u0,tspan,[0])
sol = solve(prob, saveat=ts)
dat_intrinsic_guess = Array(sol)


#####
##### Initialize the controller and then learn the control signal
#####
# SRA: Learn a neural network to approximate the guess
nn_dim = 64
    U_dim = 1 # Expect that the true dimensionality is 1
# Note: need to initialize the first layer so that the whole space is used
ctr_dyn = Chain(Dense(1,nn_dim, initb=(x)-> tspan[2].*rand(x)),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, nn_dim, σ),
                Dense(nn_dim, U_dim),
                Dense(U_dim,2)) # Equivalent to the B matrix
ps = Flux.params(ctr_dyn)
    loss_U() = sum(abs2,ctr_dyn(collect(ts)') .- U_guess)
    loss_to_use = loss_U

# Initial fast learning
tol = 300
    dat_dummy = Iterators.repeated((), 100)
    opt = ADAM(0.01)
while loss_to_use() > 5*tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
end
opt = ADAM(0.001)
while loss_to_use() > tol
    Flux.train!(loss_to_use, ps, dat_dummy, opt, cb = ()->@show loss_to_use())
end
println("Finished learning control signal")

#####
##### Initialize the intrinsic dynamics
#####
core_dyn = Dense(param(Float32.(A[:, 1:n])),
                 param(Float32.(A[:, end])),
                 identity)

#####
##### Data: Save separate predictions
#####

# Intrinsic dynamics
function dudt1_(u::AbstractArray,p,t)
    core_dyn(u)
end
u0 = dat[:,1]
prob = ODEProblem(dudt1_,u0,tspan,[0])
sol = Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=_x, saveat=ts))
dat_intrinsic_NN = Array(Flux.data(sol))

# Controllers
dat_controller_NN = Flux.data(ctr_dyn(collect(ts)'))




#####
##### PLOT1: separate training
#####
# p1 = plot(dat_intrinsic_guess[2,:], legend=false, c=COLOR_DICT["intrinsic"], lw=3);
#     title!("Approximate f(x)");
# p2 = plot(U_guess[2,:], legend=false, c=COLOR_DICT["control_space"], lw=3);
#     title!("Approximate h(t)");

p3 = plot(dat_intrinsic_NN[2,:], legend=false, c=COLOR_DICT["intrinsic"], lw=3);
    title!("NN for f(x)");
p4 = plot(dat_controller_NN[2,:], legend=false, c=COLOR_DICT["control_space"], lw=3);
    title!("NN for h(t)");

# Full plot
# my_layout = @layout [[p1 p3]; [p2 p4]];
#     p = plot(p1, p3, p2, p4, layout = my_layout)
my_layout = @layout [p3; p4];
    p = plot(p3, p4, layout = my_layout)

fname = FIGURE_FOLDERNAME * "fig_flowchart2_separateNN.png";
savefig(p, fname)


#####
##### SOLVE the initialized mixed NODE
#####
p = param(Float32[1.0; 1.0])

function dudt_(u::AbstractArray,p,t)
    core_dyn(u) + p.*ctr_dyn([t])
end
prob = ODEProblem(dudt_,u0,tspan,p)

_x = param(u0) # The initial condition can be learned too

function predict_rd()
    Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=_x, saveat=ts))
end
function loss_rd(λ1=0.01, λ2=0.1)
    return sum(abs2,predict_rd() .- dat) +      # Good fit
            λ1*loss_U() +                # Control signal close to guess
            λ2*sum(abs,ctr_dyn(collect(ts)'))   # L1 norm on ctr
end
ps = params(core_dyn,ctr_dyn,p);

# DATA: save the initalized (untrained) NN
dat_untrained = Flux.data(predict_rd())

# Initial quick learning
opt = ADAM(0.0001)
dat_dummy = Iterators.repeated((), 10)
Flux.train!(loss_rd, ps,
            dat_dummy, opt, cb = ()->@show loss_rd())
# Run for a long time
opt = ADAM(0.00001)
tol = 50
dat_dummy = Iterators.repeated((), 10)
while loss_rd() > tol
    Flux.train!(loss_rd, ps,
                dat_dummy, opt, cb = ()->@show loss_rd())
    break
end

# Save post-training fit
dat_trained = Flux.data(predict_rd())

#####
##### PLOT2: Full training
#####
# p1 = plot(dat_untrained[2,:], label="Prediction", c=COLOR_DICT["prediction"], lw=3);
#     plot!(dat[2,:], label="Data", c=COLOR_DICT["data"])
#     title!("Pre-training");
#
# p2 = plot(dat_trained[2,:], legend=false, c=COLOR_DICT["prediction"], lw=3);
#     plot!(dat[2,:], c=COLOR_DICT["data"])
#     title!("Post-training");

p1 = plot(dat_trained[2,:], label="Prediction", c=COLOR_DICT["prediction"], lw=3);
    plot!(dat[2,:], label="Data", c=COLOR_DICT["data"])
    title!("Final model");

# Full plot
# my_layout = @layout [p1 p2];
#     p = plot(p1, p2, layout = my_layout)
p = p1;

fname = FIGURE_FOLDERNAME * "fig_flowchart3_final.png";
savefig(p, fname)
