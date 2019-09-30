using PkgSRA, Plots
using BSON: @load
pyplot()

#####
##### Load variables
#####
this_dat_name = DAT_FOLDERNAME*"dat_flowchart_inkscape_"

# Original ODE and controller variables
fname = this_dat_name*"ode_vars.bson"
@load fname dat dat_grad true_grad U_true

# Bayesian variables 1: before control
fname = this_dat_name*"naive_vars.bson";
@load fname sample_trajectory_mean sample_trajectory_noise residual ctr_guess

# Bayesian variables 1: after control
fname = this_dat_name*"ctr_vars.bson";
@load fname sample_trajectory_mean_ctr sample_trajectory_noise_ctr


#####
##### Plot everything
#####
this_fig_name = FIGURE_FOLDERNAME*"TMP_fig_flowchart_inkscape_"

## 1:  3d example plot: data
plot_data = plot3d(dat[1, :], dat[2, :], dat[3, :],
        color=COLOR_DICT["true"], lw=2, legend=false,
        titlefontsize=32)
        title!("Data")

fname = this_fig_name * "data.png";
savefig(plot_data, fname)


## 2: 3d example plot: uncontrolled
uncontrolled_ind = 1:200;
    ind = uncontrolled_ind;
plot_uncontrolled = plot3d(
        all_vals[1, ind, 1], all_vals[1, ind, 2], all_vals[1, ind, 3],
        color=COLOR_DICT["model_uncontrolled"], alpha=0.8, lw=2,
        legend=false)
    plot3d!(dat[1, ind], dat[2, ind], dat[3, ind],
            color=COLOR_DICT["true"], lw=2)
    title!("Naive Model", titlefontsize=28)

fname = this_fig_name * "uncontrolled.png";
savefig(plot_uncontrolled, fname)


## 3: Residual with noise envelope (z coordinate)
plot_coordinate = 3;
residual_ind = 1:200;
    ind = residual_ind;
plot_residual = plot(residual[plot_coordinate,ind],
                    ribbon=mean(all_noise), fillalpha=0.5,
                    color=COLOR_DICT["residual"], legend=false,
                    xticks=false, yticks=false, lw=3)
    title!("Residual", titlefontsize=28)

fname = this_fig_name * "residual.png";
savefig(plot_residual, fname)


## 4: Control signal guess (z coordinate)
# Same ind as above
plot_ctr_guess = plot(ctr_guess[plot_coordinate,ind],
                    color=COLOR_DICT["control_time"], legend=false,
                    xticks=false, yticks=false, lw=3)
    title!("Control Signal Guess", titlefontsize=28)

fname = this_fig_name * "control_guess.png";
savefig(plot_ctr_guess, fname)


## 5: Controlled model
controlled_ind = 1:500;
    ind = controlled_ind;
    ctr_dat = trajectory_samples_ctr[1];
plot_controlled = plot3d(
        ctr_dat[1, ind], ctr_dat[2, ind], ctr_dat[3, ind],
        color=COLOR_DICT["model_controlled"], alpha=0.8, lw=2,
        legend=false)
    plot3d!(dat[1, ind], dat[2, ind], dat[3, ind],
            color=COLOR_DICT["true"], lw=2)
    title!("Controlled Model", titlefontsize=28)

fname = this_fig_name * "controlled.png";
savefig(plot_controlled, fname)
