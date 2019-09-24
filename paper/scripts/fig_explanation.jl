using PkgSRA
using Plots, Random
pyplot()
# pgfplots() # DOESN'T WORK
Random.seed!(11)
include("paper_settings.jl");

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("../examples/example_mass_spring.jl")

# Define the time dependent forcing function
# num_ctr = 2;
# U_starts = rand(num_ctr,1) .* tspan[2]
# U_width = 0.05;
# U_func(t) = U_func_time(t, U_width, U_starts)
U_func = spring_forcing_example(tspan)


# Solve the system
dyn_with_ctr = solve_msd_system(U_func)
dyn_no_ctr = solve_msd_system()
dyn_only_ctr = U_func.(collect(ts'))

#####
##### Produce the plots
#####
# First: intrinsic dynamics
p1 = plot(dyn_no_ctr[1,:], label="y");
    plot!(dyn_no_ctr[2,:], label="v");
    title!("Intrinsic Dynamics");

# Second: Controller
p2 = plot(dyn_only_ctr');
    title!("Control signal");

# Third: controlled system
p3 = plot(dyn_with_ctr[1,:], label="y");
    plot!(dyn_with_ctr[2,:], label="v");
    title!("Controlled Dynamics");

# Creat the layout and plot
my_layout = @layout [[p1; p2] p3];
    p = plot(p1, p2, p3, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_explanation.png";
savefig(p, fname)
