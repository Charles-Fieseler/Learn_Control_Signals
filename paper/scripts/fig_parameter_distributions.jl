using PkgSRA, Plots, LinearAlgebra, Statistics, Turing, Distributions
using AxisArrays, DataStructures
using BSON: @load
pyplot()

#####
##### Load variables
#####
# Load true model
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

include("paper_settings.jl")
this_dat_name = DAT_FOLDERNAME*"TMP_dat_flowchart_inkscape_"

# Intermediate data
fname = this_dat_name*"intermediate.bson";
@load fname chain chain_sub
# SINDy variables 1: before control
fname = this_dat_name*"naive_vars_sindy.bson";
@load fname unctr_nz_terms unctr_nz_names
# SINDY variables 2: after control
fname = this_dat_name*"ctr_vars_sindy.bson";
@load fname sub_nz_terms sub_nz_names

#####
##### Plot
#####

a = sample(chain[:noise], 1000)
b = sample(chain_sub[:noise], 1000)

histogram(Array.([a,b]), bar_width=0.5)

histogram(a, bar_width=0.5)
    histogram!(b)

d1 = chain[:noise].value.data[:,1,1]
plot_noise = histogram(d1, legend=false)
d2 = chain_sub[:noise].value.data[:,1,1]
width = plot_noise.series_list[1].plotattributes[:bar_width]
histogram!(d2, legend=false, bar_width=width[1], nbins=1)


plot_noise.series_list[1].plotattributes[:bar_width]


plot_var = "all_coef[1]"

# d1 = chain[var].value.data[:,1,1]
plot_noise = density(chain[plot_var], legend=false)
density!(chain_sub[plot_var], legend=false)

mixeddensity(chain_sub)
