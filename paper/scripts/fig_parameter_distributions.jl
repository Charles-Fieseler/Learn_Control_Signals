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
@load fname sindy_unctr
# SINDY variables 2: after control
fname = this_dat_name*"ctr_vars_sindy.bson";
@load fname sindy_sub

#####
##### Plot
#####
get_nonzero_term_names()
