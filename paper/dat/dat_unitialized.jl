using PkgSRA
using Plots, Random
using Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
using DSP
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("paper_settings.jl")
include(EXAMPLE_FOLDERNAME*"example_mass_spring.jl")

# Define the time dependent forcing function
U_func = spring_forcing_example(tspan)

# Solve the system
dat = Array(solve_msd_system(U_func))
dat_unforced = Array(solve_msd_system())
# Add a constant to the data
n, m = size(dat)
dat_and_const = vcat(dat, ones(1,m))

# Get true control signal
U_true = zeros(size(dat))
for (i, t) in enumerate(collect(ts))
    U_true[:,i] = U_func(t, dat[:,1])
end

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
##### Try to learn the dynamics directly
#####
nn_dim = 64
    U_dim = 1 # Expect that the true dimensionality is 1
core_dyn_naive = Dense(n, n, identity)
# Part 2: Neural network for control signal
ctr_dyn_naive = Chain(Dense(1,nn_dim, initb=(x)-> tspan[2].*rand(x)),
                        Dense(nn_dim, nn_dim, σ),
                        Dense(nn_dim, nn_dim, σ),
                        Dense(nn_dim, U_dim),
                        Dense(U_dim,2)) # Equivalent to the B matrix
p = param(Float32[1.0; 1.0])

function dudt_naive(u::AbstractArray,p,t)
    core_dyn_naive(u) + p.*ctr_dyn_naive([t])
end
prob_naive = ODEProblem(dudt_naive,u0,tspan,p)

_x = param(u0) # The initial condition can be learned too

function predict_rd_naive()
    Flux.Tracker.collect(diffeq_rd(p,prob_naive,Tsit5(),u0=_x, saveat=ts))
end
function loss_naive(λ=0.01)
    return sum(abs2,predict_rd_naive() .- dat) +      # Good fit
            λ*sum(abs,ctr_dyn_naive(collect(ts)'))   # L1 norm on ctr
end

# SOLVE the mixed NODE
ps = params(core_dyn_naive, ctr_dyn_naive, p);

# Initial quick learning
opt = ADAM(0.01)
dat_dummy = Iterators.repeated((), 300)
Flux.train!(loss_naive, ps,
            dat_dummy, opt, cb = ()->@show loss_naive())
# Run for a long time
opt = ADAM(0.001)
tol = 25
dat_dummy = Iterators.repeated((), 20)
num_iter = 1
max_iter = 10
while loss_naive() > tol
    Flux.train!(loss_naive, ps,
                dat_dummy, opt, cb = ()->@show loss_naive())
    global num_iter += 1
    num_iter > max_iter && break
end


#####
##### Plot and save: uninitialized
#####
# Full fit
sol = Flux.data(predict_rd_naive())
p_naive_full = plot(sol[2,:], label="Simulation",
                    color=COLOR_DICT["intrinsic"], lw=2);
    plot!(dat_and_const[2,:], label="Data",
                    color=COLOR_DICT["data"], lw=3);
    title!("Unitilialized Fit");

# Only "dynamics"
function dudt_naive_only_dyn(u::AbstractArray,p,t)
    core_dyn_naive(u)
end
prob_naive_only_dyn = ODEProblem(dudt_naive_only_dyn,u0,tspan,p)
function predict_rd_naive_dyn()
    Flux.Tracker.collect(diffeq_rd(p,prob_naive_only_dyn,Tsit5(),u0=_x, saveat=ts))
end
p_naive_dyn = plot(Flux.data(predict_rd_naive_dyn())[2,:], legend=false,
                    color=COLOR_DICT["intrinsic"], lw=2);
    title!("Intrinsic Dynamics");

# Only "control"
dat_naive_ctr = Flux.data(ctr_dyn_naive(collect(ts)'))
p_naive_ctr = plot(dat_naive_ctr[2,:], legend=false,
                    color=COLOR_DICT["control_time"], lw=2);
    title!("Control Signal");

# Save everything
my_layout = @layout [p_naive_full  [p_naive_dyn; p_naive_ctr]];
    p_all = plot(p_naive_full, p_naive_dyn, p_naive_ctr, layout = my_layout)
fname = FIGURE_FOLDERNAME * "fig_uninitialized.png";
savefig(p_all, fname)



#####
##### Instead: initialize the controller and then learn
#####
# SRA: Learn a neural network to approximate the guess
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
tol = 500
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
# plot(Flux.data(ctr_dyn(collect(ts)')[2,:]))

## Now learn entire system
core_dyn = Dense(param(Float32.(A[:, 1:n])),
                 param(Float32.(A[:, end])),
                 identity)
# Part 2: Neural network for control signal; initialized above
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

## SOLVE the initialized mixed NODE
ps = params(core_dyn,ctr_dyn,p);

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


#####
##### Plot and save: Initialized
#####
# Full fit
sol = Flux.data(predict_rd())
p_full = plot(dat_and_const[2,:], label="Data",
                            color=COLOR_DICT["data"], lw=3);
    plot!(sol[2,:], label="Simulation",
                            color=COLOR_DICT["intrinsic"], lw=2);
    title!("Initialized Fit");

# Only "dynamics"
function dudt_only_dyn(u::AbstractArray,p,t)
    core_dyn(u)
end
prob_only_dyn = ODEProblem(dudt_only_dyn,u0,tspan,p)
function predict_rd_dyn()
    Flux.Tracker.collect(diffeq_rd(p,prob_only_dyn,Tsit5(),u0=_x, saveat=ts))
end
p_dyn = plot(dat_unforced[2,:], label="True",
                    color=COLOR_DICT["intrinsic_true"], lw=3)
        plot!(Flux.data(predict_rd_dyn())[2,:], label="Simulation",
                    color=COLOR_DICT["intrinsic"], lw=2);
    title!("Intrinsic Dynamics");

# Only "control"
dat_full_ctr = Flux.data(p.*ctr_dyn(collect(ts)'))
p_ctr = plot(U_true[2,:], label="True", lw=2,
            color=COLOR_DICT["control_true"])
        plot!(dat_full_ctr[2,:], label="Simulation",
                color=COLOR_DICT["control_time"], lw=2);
    title!("Control Signal");

# Save everything
my_layout = @layout [p_full  [p_dyn; p_ctr]];
    p_all = plot(p_full, p_dyn, p_ctr, layout = my_layout)
fname = FIGURE_FOLDERNAME * "fig_initialized.png";
savefig(p_all, fname)
