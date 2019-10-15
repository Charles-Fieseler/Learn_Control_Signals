using PkgSRA, Plots, LaTeXStrings
using Turing, AxisArrays, DataFrames, LinearAlgebra, DataStructures
using BSON: @load
pyplot()


#####
##### Load the data
#####
this_dat_name = DAT_FOLDERNAME*"dat_neuron_"

## Constant input
fname = this_dat_name*"controlled_model.bson";
@load fname chain_ctr

## Spikes with varying input
fname = this_dat_name*"controlled_model2.bson";
@load fname chain_ctr2

#####
##### Produce the plots
#####
plot_opt = Dict(:titlefontsize=>24,
        :yticks=>false, :fontfamily=>:serif,
        :legendfontsize=>14, :guidefontsize=>16,
        :xtickfontsize=>16)

# Plot
key = "all_coef[1]"
plot1 = density(Array(chain_ctr[key]), lw=3; plot_opt...)
        density!(Array(chain_ctr2[key]), lw=3)
        vline!([5], lw=3, color=:black, legend=false)
        title!("Coefficient for v")

key = "all_coef[5]"
plot2 = density(Array(chain_ctr[key]), lw=3; plot_opt...)
        density!(Array(chain_ctr2[key]), lw=3)
        vline!([180], lw=3, color=:black, legend=false)
        title!("Constant term")

key = "all_coef[7]"
plot3 = density(Array(chain_ctr[key]), lw=3, label="Constant Input"; plot_opt...)
        density!(Array(chain_ctr2[key]), label="Variable Input", lw=3)
        vline!([0.04], lw=3, color=:black,
                label="True value", legend=:bottomright)
        xlims!(0.03, 0.05)
        title!(L"v^2")

# Create the layout and plot
my_layout = @layout [p1; p2; p3];
    p_final = plot(plot1, plot2, plot3, layout = my_layout)

# Save
fname = FIGURE_FOLDERNAME * "fig_neuron_parameters.png";
savefig(p_final, fname)
