using PkgSRA, Plots, LinearAlgebra, Statistics
using BSON: @load
pyplot()

#####
##### Load variables
#####
include("paper_settings.jl")
this_dat_name = DAT_FOLDERNAME*"TMP_dat_flowchart_inkscape_"

# Uncontrolled data
fname = this_dat_name*"uncontrolled_lorenz.bson"
@load fname dat_raw numerical_grad_raw

# Original ODE and controller variables
fname = this_dat_name*"ode_vars.bson";
@load fname dat dat_grad true_grad U_true

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars_sindy.bson";
@load fname sindy_dat_unctr sindy_unctr

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars_bayes.bson";
@load fname dat_unctr sample_trajectory_noise residual accepted_ind ctr_guess

# SINDY variables 2: after control
fname = this_dat_name*"ctr_vars_sindy.bson";
@load fname sindy_sub sindy_dat_ctr sindy_grad_ctr

# Bayesian variables 2: after control
fname = this_dat_name*"ctr_vars_turing.bson";
@load fname dat_ctr sample_trajectory_noise_ctr sample_gradients_ctr turing_dat_ctr ctr_guess2


#####
##### Plot everything
#####
this_fig_name = FIGURE_FOLDERNAME*"TMP_fig_flowchart_inkscape_"
plot_opt = Dict(:titlefontsize=>48,
        :xticks=>false, :yticks=>false, :zticks=>false,
        :legend=>false, :fontfamily=>:serif)

## 0: Raw, uncontrolled system
plot_raw = plot3d(dat_raw[1, :], dat_raw[2, :], dat_raw[3, :],
        color=COLOR_DICT["data_uncontrolled"], lw=4; plot_opt...)
        title!("Unforced Lorenz")
fname = this_fig_name * "uncontrolled.png";
savefig(plot_raw, fname)
