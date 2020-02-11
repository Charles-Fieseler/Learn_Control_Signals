using PkgSRA, Plots, LinearAlgebra, Statistics
using AxisArrays, DataFrames, DataStructures
using BSON: @load
pyplot()

include("paper_plotting.jl")
include("paper_settings.jl")

#####
##### Plot AND save using external function
#####
this_dat_name = "dat_library_of_examples_lv_noise_"
plot_final = plot_library_noise(this_dat_name, "Lotka Volterra";
                        plot_opt=plot_opt)

this_fig_name = FIGURE_FOLDERNAME*"fig_library_with_noise";
fname = this_fig_name * "lotkaVolterra.png";
savefig(plot_final, fname)
