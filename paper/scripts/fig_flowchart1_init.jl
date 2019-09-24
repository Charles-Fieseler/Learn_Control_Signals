using PkgSRA
using Plots, Random
using LaTeXStrings
pyplot()
Random.seed!(11)
include("paper_settings.jl");

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
dyn_with_ctr = solve_msd_system(U_func)
dyn_only_ctr = U_func.(collect(ts'))

#####
##### Get the SRA intermediate products
#####
dat = dyn_with_ctr
n, m = size(dat)
dat = vcat(dat, ones(1,m))

dat_grad = numerical_derivative(dat, ts)

# Prediction for uncontrolled data
dat_grad_guess = (dat_grad/dat)*dat

# Actual SRA
all_U = []
U_guess = sra_dmd(dat_and_const, dat_grad,
                    quantile_threshold=0.9)[1]
push!(all_U, U_guess)
A, B = dmdc(dat_and_const, dat_grad, U_guess)
# Loop a couple of times
for i in 1:5
    pt = plot(U_guess[2,:], label="Residual")
    plot!(U_func.(ts), label="True")
    title!("Iteration $i")
    display(pt)
    # Update guess for control signal
    global U_guess = calculate_residual(dat_and_const,
                                        dat_grad,
                                        A)
    push!(all_U, U_guess)
    sparsify_signal!(U_guess, quantile_threshold=0.9)
    # Update guess for dynamics
    global A, B = dmdc(dat_and_const, dat_grad, U_guess)
end

#####
##### Produce the plots
#####
# First: Just data
p1 = plot(dat_grad[2,:], legend=false, linecolor=COLOR_DICT["data"], lw=3);#, label="Data");
    # plot!(dyn_only_ctr[1,:], label="Control signal");
    title!("Derivative");
    # annotate!(30.0, 2.0, "ẋ = Ax + BÛ");

# Second: Initial residual
i = 1;
p21 = plot(all_U[i][2,:], legend=false, linecolor=COLOR_DICT["control_time"], lw=3);
    title!("Residual: i=$i");
i = 2;
p22 = plot(all_U[i][2,:], legend=false, linecolor=COLOR_DICT["control_time"], lw=3);
    title!("Residual: i=$i");
i = 3;
p23 = plot(all_U[i][2,:], legend=false, linecolor=COLOR_DICT["control_time"], lw=3);
    title!("Residual: i=$i");

# Creat the layout and plot
my_layout = @layout [p1 [p21; p22; p23]];
    p = plot(p1, p21, p22, p23, layout = my_layout)
    # suptitle("Initialization");
    display(p)

# Save
fname = FIGURE_FOLDERNAME * "fig_flowchart1_init.png";
savefig(p, fname)
