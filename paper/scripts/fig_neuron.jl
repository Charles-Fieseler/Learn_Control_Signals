using PkgSRA, Plots, StatsPlots
using Turing, AxisArrays, DataFrames, LinearAlgebra, DataStructures
using BSON: @load
pyplot()


#####
##### Load the data
#####
this_dat_name = DAT_FOLDERNAME*"dat_neuron_"

## Basic spiking model
# Raw data
fname = this_dat_name*"raw.bson";
@load fname dat grad_true numerical_grad U_true

# Controlled model
fname = this_dat_name*"controlled_model.bson";
@load fname ctr_final 

## Spikes with varying input
# Raw data
fname = this_dat_name*"raw2.bson";
@load fname dat2 grad_true2 numerical_grad2 U_true2

# Controlled model
fname = this_dat_name*"controlled_model2.bson";
@load fname ctr_final2

#####
##### Produce the plots
#####
plot_opt = Dict(:titlefontsize=>24,
        :yticks=>false, :fontfamily=>:serif,
        :legendfontsize=>16)

## Two density plots
# Two panels: data, then control
plot_data = plot(ts, dat[1,:], lw=5,
                color=COLOR_DICT["data"],
                legend=false, xticks=false;
                plot_opt...);
    xlabel!("");
    title!("Neuron With Constant Input ")

# Second: Controller
plot_control = plot(ts, ctr_final[1,:], legend=false,
                color=COLOR_DICT["control_time"], lw=3; plot_opt...);
    xlabel!("Time", guidefontsize=14, tickfontsize=14);
    title!("Learned Controller")

# Create the layout and plot
my_layout = @layout [p1; p2];
    p_final = plot(plot_data, plot_control, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_neuron.png";
savefig(p_final, fname)


##
##
## Part 2: with external control changes
plot_data2 = plot(ts, dat2[1,:], lw=5,
                color=COLOR_DICT["data"],
                legend=false, xticks=false;
                plot_opt...);
    xlabel!("");
    title!("Neuron With Varying Input")

# Second: Learned and true Controller
inset_ind = 3200:4200; # Chosen by hand

plot_control2 = plot(ts, ctr_final2[1,:], legend=false,
                color=COLOR_DICT["control_time"], lw=3; plot_opt...);
    xlabel!("Time", guidefontsize=14, tickfontsize=14);
    title!("Learned Controller")
    plot!([ts[inset_ind[1]], ts[inset_ind[1]]],
        [-1000, 1000], color=:black, lw=3)
    plot!([ts[inset_ind[end]], ts[inset_ind[end]]],
        [-1000, 1000], color=:black, lw=3)

# Create what will be an inset
plot_inset = plot(U_true2[1,inset_ind], lw=5,
                    color=COLOR_DICT["control_true"], # label="True",
                    xticks=false,
                    legendfontsize=48, legend=false) #legend=:bottomleft)
    plot!(ctr_final2[1,inset_ind], lw=5,# label="Learned",
        color=COLOR_DICT["control_time"],
        ylims=[-100, 300]; plot_opt...)

# Create the layout and plot
my_layout = @layout [p1; p2];
    p_final2 = plot(plot_data2, plot_control2, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_neuron2.png";
savefig(p_final2, fname)

fname = FIGURE_FOLDERNAME * "fig_neuron_inset.png";
savefig(plot_inset, fname)
