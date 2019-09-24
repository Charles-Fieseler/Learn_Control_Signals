using PkgSRA
using Plots, Random
pyplot()
Random.seed!(11)
include("paper_settings.jl");

#####
##### Define the controlled dynamical system
#####
# Load example problem
include(EXAMPLE_FOLDERNAME*"example_falling_ball.jl")
U_starts = [6.0]
    U_widths = 0.2;
    amplitude = 50.0;
U_func_kick(t, u) = U_func_time(t, u,
                                U_widths, U_starts,
                                F_dim=2,
                                amplitude=amplitude)

# Define the time dependent forcing function

U_func_wall(X) = U_func_spring(X, k=1e3, r=1.0);
# U_func_wall = spring_forcing_example(tspan)


# Solve the system
dyn_with_ctr = solve_ball_system(U_func_space=U_func_wall,
                                 U_func_time=U_func_kick)

# plot(dyn_with_ctr)

dat = Array(dyn_with_ctr)
dyn_control_wall = zeros(size(dat))
for i in 1:size(dat, 2)
    dyn_control_wall[:,i] = U_func_wall(dat[:, i])
end

dyn_control_kick = zeros(size(dat))
for (i, t) in enumerate(ts)
    dyn_control_kick[:,i] = U_func_kick(t, dat[:, i])
end

# dyn_no_ctr = solve_ball_system()

#####
##### Produce the plots
#####
# First: intrinsic dynamics
plot_data = plot(ts, dyn_with_ctr[1,:], label="Height", lw=3, legend=:bottomright);
    plot!(ts, dyn_with_ctr[2,:], label="Velocity", lw=3,
            legendfontsize=14, xticks=false, yticks=false);
    xlabel!("");
    title!("Observable Data", titlefontsize=24);

# Second: Controller
plot_control = plot(ts, vec(dyn_control_wall[2,:]), label="Ground",
                color=COLOR_DICT["control_true"], lw=3);
    plot_kick = plot!(ts, vec(dyn_control_kick[2,:]), label="Kick",
                color=COLOR_DICT["control_time"], lw=3,
                legendfontsize=14, xticks=false, yticks=false);
    xlabel!("Time", fontsize=16, legend=:bottomright);
    title!("Hidden Control Signals", titlefontsize=24);

# Creat the layout and plot
my_layout = @layout [p1; p2];
    p_final = plot(plot_data, plot_control, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_ball_intrinsic_and_control.png";
savefig(p_final, fname)
