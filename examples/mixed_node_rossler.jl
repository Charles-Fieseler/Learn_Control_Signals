using PkgSRA
using Plots, Random
using Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("../examples/example_rossler.jl")

# Define the forcing functions
num_ctr = 20;
U_starts = rand(num_ctr,1) .* tspan[2]
U_width = 0.2;
my_U_func_time(t, u) = U_func_time(t, u, U_width, U_starts, F_dim=3)

# my_U_func_space(u) = U_func_spring(u, r=-10.0, k=100.0, damping=0.001,
#                                    F_dim=3, damp_dim=1)
my_U_func_space(u) = U_func_spring_sphere(u, r=5.0, location=[-15.0,0.0,0.0],
                                          k=1e5, damping=1e-5,
                                          F_dim=3, damp_dim=3)

# Solve and plot the raw system
dat = solve_rossler_system()
plot3d(dat[1,:], dat[2,:], dat[3,:])

# Solve and plot the time forced system
dat_ft = solve_rossler_system(my_U_func_time)
plot3d(dat_ft[1,:], dat_ft[2,:], dat_ft[3,:])
xlabel!("x"); ylabel!("y")

# Solve and plot the spatially forced system
dat_fx = solve_rossler_system(U_func_time_trivial,
                              my_U_func_space)
plot3d(dat_fx[1,:], dat_fx[2,:], dat_fx[3,:])

# Solve and plot the doubly-forced system
dat_fxt = solve_rossler_system(my_U_func_time,
                              my_U_func_space)
plot3d(dat_fxt[1,:], dat_fxt[2,:], dat_fxt[3,:])
# plot(dat_fxt[1,:], dat_fxt[2,:])


#####
##### Solve a basic SINDy model on the unforced data
#####
dat_grad = numerical_derivative(dat, ts)
sindy_terms = Dict("cross_terms"=>2, "constant"=>nothing);
model = sindyc(dat, dat_grad, library=sindy_terms)

# First visualize the gradients
plot(dat_grad[1,:], label="data")
   plot!(model(dat)[1,:], label="sindy")

# Solve the differential equation
u0 = dat[:,1]
prob = ODEProblem(model,u0,tspan,[0.0])
sol = solve(prob, Tsit5(), saveat=ts)

dat_sindy = Array(sol)
plot3d(dat_sindy[1,:], dat_sindy[2,:], dat_sindy[3,:])
   plot3d!(dat[1,:], dat[2,:], dat[3,:])
