using PkgSRA, Plots, LinearAlgebra, Statistics
using AxisArrays, DataFrames, DataStructures
using BSON: @load
pyplot()

include("paper_plotting.jl")
include("paper_settings.jl")

#####
##### Plot AND save using external function
#####

## Lotka Volterra
this_dat_name = "dat_library_with_noise_lv_"
plot_final = plot_library_noise(this_dat_name, "Lotka Volterra";
                        plot_opt=plot_opt)

this_fig_name = FIGURE_FOLDERNAME*"fig_library_with_noise_";
fname = this_fig_name * "lotkaVolterra.png";
savefig(plot_final, fname)

## FitzHugh Nagumo
this_dat_name = "dat_library_with_noise_fhn_"
plot_final = plot_library_noise(this_dat_name, "FitzHugh Nagumo";
                        plot_opt=plot_opt)

this_fig_name = FIGURE_FOLDERNAME*"fig_library_with_noise_";
fname = this_fig_name * "fitzHughNagumo.png";
savefig(plot_final, fname)
