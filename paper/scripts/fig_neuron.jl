using PkgSRA, Plots
using Turing, AxisArrays, DataFrames
using BSON: @load
pyplot()


#####
##### Load the data
#####
this_dat_name = DAT_FOLDERNAME*"dat_neuron_"

# Raw data
fname = this_dat_name*"raw.bson";
@save fname dat true_grad numerical_grad

# Controlled model
fname = this_dat_name*"controlled_model.bson";
@save fname ctr_final

#####
##### Produce the plots
#####
plot_opt = Dict(:titlefontsize=>24,
        :yticks=>false, :fontfamily=>:serif,
        :legendfontsize=>16)

## Figure in two panels: data, then control
plot_data = plot(ts, dat[1,:], lw=5,
                legend=false, xticks=false;
                plot_opt...);
    xlabel!("");
    title!("Spiking Neuron")

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
