using PkgSRA, Plots
using Turing, AxisArrays, DataFrames
using BSON: @load
pyplot()


#####
##### Load the data
#####
this_dat_name = DAT_FOLDERNAME*"dat_ball_"

# Raw data
fname = this_dat_name*"raw.bson";
@load fname dat numerical_grad dyn_control_kick dyn_control_wall

# Naive model
fname = this_dat_name*"naive_model.bson";
@load fname dat_naive naive_model

# Controlled model
fname = this_dat_name*"controlled_model.bson";
@load fname chain_ctr ctr_final dat_final

#####
##### Produce the plots
#####
plot_opt = Dict(:titlefontsize=>48,
        :xticks=>false, :yticks=>false, :fontfamily=>:serif,
        :legendfontsize=>16, :titlefontsize=>24)

## First figure in two panels: intrinsic dynamics

# First: intrinsic dynamics
plot_data = plot(ts, dat[1,:], label="Height", lw=5,
                legend=:topleft);
    plot!(ts, dat[2,:], label="Velocity", lw=5;
    plot_opt...);
    xlabel!("");
    title!("Observable Data")

# Second: Controller
plot_control = plot(ts, vec(dyn_control_wall[2,:]), label="Ground",
                color=COLOR_DICT["control_true"], lw=3);
    plot!(ts, vec(dyn_control_kick[2,:]), label="Kick",
            color=COLOR_DICT["data"], lw=3,
            legendfontsize=14, xticks=false, yticks=false,
            legend=:topright; plot_opt...);
    xlabel!("Time", fontsize=24);
    title!("Hidden Control Signals")

# Create the layout and plot
my_layout = @layout [p1; p2];
    p_final = plot(plot_data, plot_control, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_ball_intrinsic_and_control.png";
savefig(p_final, fname)



## Second figure in three panels: Model dynamics

# First, naive model dynamics
plot_naive = plot(ts, dat_naive[1,:], lw=5,
                legend=:topright);
    plot!(ts, dat_naive[2,:], legend=false, lw=5;
        plot_opt...);
    xlabel!("");
    title!("Naive Model")

# Second, learned control signals
plot_ctr = plot(ts, ctr_final[2,:], lw=5,
                color=COLOR_DICT["control_time"], legend=false;
                plot_opt...);
    xlabel!("");
    title!("Learned Controller")


# Third, Reconstruction of controlled model
plot_final = plot(ts, dat_final', lw=5,
                # color=COLOR_DICT["model_controlled"],
                legend=false;
                plot_opt...);
    xlabel!("Time", guidefontsize=18);
    title!("Controlled Model")

# Create the layout and plot
my_layout = @layout [plot_naive; plot_ctr; plot_final];
    p_final = plot(plot_naive, plot_ctr, plot_final, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_ball_model.png";
savefig(p_final, fname)
