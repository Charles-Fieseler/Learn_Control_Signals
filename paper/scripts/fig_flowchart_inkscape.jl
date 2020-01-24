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

## 1:  3d example plot: data
plot_data = plot3d(dat[1, :], dat[2, :], dat[3, :],
        color=COLOR_DICT["true"], lw=3; plot_opt...)
        title!("Perturbed Data")
fname = this_fig_name * "data.png";
savefig(plot_data, fname)

## 2: Two 1d example plots: naive model
uncontrolled_ind = 1:500;
    ind = uncontrolled_ind;
let d = sindy_dat_unctr
        global plot_uncontrolled1 = plot(d[1, ind],
                color=COLOR_DICT["model_uncontrolled"],
                alpha=0.8, lw=4;
                plot_opt...)
end
    plot!(dat[1, ind], color=COLOR_DICT["true"], lw=4)
    ylabel!("X", guidefontsize=34)
    title!("Naive Model")
let d = sindy_dat_unctr
    global plot_uncontrolled2 = plot(d[2, ind],
            color=COLOR_DICT["model_uncontrolled"],
            alpha=0.8, lw=4;
            plot_opt...)
end
    plot!(dat[2, ind], color=COLOR_DICT["true"], lw=4)
    ylabel!("dY/dt", guidefontsize=34)

fname = this_fig_name * "uncontrolledX.png";
savefig(plot_uncontrolled1, fname)
fname = this_fig_name * "uncontrolledY.png";
savefig(plot_uncontrolled2, fname)


## 3: Residual with noise envelope (x and y coordinates)
plot_coordinate = 1;
    residual_ind = 1:500;
    ind = residual_ind;
plot_residual1 = plot(residual[plot_coordinate,ind],
                    ribbon=mean(sample_trajectory_noise),
                    fillalpha=0.5,
                    color=COLOR_DICT["residual"], lw=4;
                    plot_opt...)
    ylabel!("dX/dt", guidefontsize=34)
    title!("Residual")
plot_coordinate = 2;
plot_residual2 = plot(residual[plot_coordinate,ind],
                ribbon=mean(sample_trajectory_noise),
                fillalpha=0.5,
                color=COLOR_DICT["residual"], lw=4;
                plot_opt...)
    ylabel!("dY/dt", guidefontsize=24)

fname = this_fig_name * "residualX.png";
savefig(plot_residual1, fname)
fname = this_fig_name * "residualY.png";
savefig(plot_residual2, fname)


## 4: Subsampling
plot_coordinate = 1;
    sub_ind = 1:500;
    ind = sub_ind;
plot_subsample = plot(dat[plot_coordinate,ind],
                    color=COLOR_DICT["true"], lw=4;
                    plot_opt...)
    ylabel!("X", guidefontsize=34)
    title!("Subsample")
    subsample_ind = intersect(ind, accepted_ind)
    scatter!(ind[subsample_ind], dat[plot_coordinate,subsample_ind],
            color=COLOR_DICT["intrinsic"],
            markershape=:o, markersize=20;
            plot_opt...)
fname = this_fig_name * "subsample.png";
savefig(plot_subsample, fname)


## 5: Partial model (same ind as above)
plot_coordinate = 1;
    ind = 1:500
plot_partial = plot(sindy_grad_ctr[plot_coordinate,ind],
                    color=COLOR_DICT["model_partial"], lw=8;
                    plot_opt...)
    plot!(dat_grad[plot_coordinate,ind],
        color=COLOR_DICT["true"], lw=4;
        plot_opt...)
    ylabel!("dX/dt", guidefontsize=34)
    title!("Partial Model")
fname = this_fig_name * "partial.png";
savefig(plot_partial, fname)


## 6: Control signal guess (x coordinate)
# Same ind as above
plot_ctr_guess = plot(ctr_guess[plot_coordinate,ind],
                    color=COLOR_DICT["control_time"], lw=6;
                    plot_opt...)
    plot!(U_true[plot_coordinate,ind],
            color=COLOR_DICT["true"], lw=3;
            plot_opt...)
    title!("Control Signal")
fname = this_fig_name * "control_guess.png";
savefig(plot_ctr_guess, fname)


## 7: "Revealed underlying system"
controlled_ind = 1:5000;
    ind = controlled_ind;
let d = sindy_dat_ctr
    global plot_controlled = plot3d(
                d[1, ind], d[2, ind], d[3, ind],
                color=COLOR_DICT["model_controlled"], alpha=0.8, lw=3;
                plot_opt...)
end
    # plot3d!(dat[1, ind], dat[2, ind], dat[3, ind],
    #         color=COLOR_DICT["true"], lw=2)
    title!("Learned System")
fname = this_fig_name * "controlled.png";
savefig(plot_controlled, fname)
