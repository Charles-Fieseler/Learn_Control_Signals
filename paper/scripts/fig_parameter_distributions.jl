using PkgSRA, Plots, LinearAlgebra, Statistics
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
@save fname chain chain_sub

#####
##### Plot
#####
