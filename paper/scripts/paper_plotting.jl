################################################################################
#####
##### Define a function for further library examples
#####

plot_opt = Dict(:titlefontsize=>28,
        :xticks=>false, :yticks=>false, :zticks=>false,
        :legend=>false, :fontfamily=>:serif)

function plot_library(this_dat_name, this_fig_name, system_name="";
                        plot_opt=plot_opt,
                        plot_ind=501:1000,
                        plot_coordinate=1,
                        var_names=["X","Y","Z"])
        ### LOAD
        fname = this_dat_name*"uncontrolled.bson"
        @load fname dat_raw #numerical_grad_raw
        fname = this_dat_name*"ode_vars.bson";
        @load fname dat U_true
        fname = this_dat_name*"naive_vars_bayes.bson";
        @load fname noise_guess residual accepted_ind #ctr_guess
        fname = this_dat_name*"ctr_vars_sindy.bson";
        @load fname dat_ctr ctr_guess2 #sindy_grad_ctr

        ### PLOTS
        ## 1: Attractor view (uncontrolled)
        if size(dat_raw,1) == 3
                plot_raw = plot3d(dat_raw[1, :], dat_raw[2, :], dat_raw[3, :],
                        color=COLOR_DICT["data_uncontrolled"], lw=4,
                        #zlabel=system_name, guidefontsize=24, mirror=true
                        ;plot_opt...);
        elseif size(dat_raw,1) == 2
                plot_raw = plot(dat_raw[1, :], dat_raw[2, :],
                        color=COLOR_DICT["data_uncontrolled"], lw=4,
                        ;plot_opt...);
        end
        title!(system_name)

        ## 2: 1d time series (controlled)
        ind = plot_ind;
        plot_1d = plot(dat[plot_coordinate, ind],
                color=COLOR_DICT["data"], lw=4; plot_opt...)
                ylabel!("$(var_names[plot_coordinate])", guidefontsize=24)
        plot_ctr_true = plot(U_true[plot_coordinate,ind],
                color=COLOR_DICT["true"], lw=3;
                plot_opt...);

        ## 3: Residual with noise envelope (x coordinate)
        plot_residual1 = plot(residual[plot_coordinate,ind],
                            ribbon=mean(noise_guess),
                            fillalpha=0.5,
                            color=COLOR_DICT["residual"], lw=4;
                            plot_opt...)
            ylabel!("d$(var_names[plot_coordinate])/dt", guidefontsize=24);

        ## 4: Control signal guess (x coordinate)
        # Same ind as above
        plot_ctr_guess = plot(ctr_guess2[plot_coordinate,ind],
                        color=COLOR_DICT["control_time"], lw=6;
                        plot_opt...)

        ## 5: "Revealed underlying system"
        controlled_ind = 1:5000;
            ind = controlled_ind;
        let d = dat_ctr
            if size(dat_ctr,1) == 3
                    global plot_reconstruction = plot3d(
                        d[1, ind], d[2, ind], d[3, ind],
                        color=COLOR_DICT["model_controlled"], alpha=0.8, lw=3;
                        plot_opt...);
            elseif size(dat_ctr,1) == 2
                    global plot_reconstruction = plot(
                        d[1, ind], d[2, ind],
                        color=COLOR_DICT["model_controlled"], alpha=0.8, lw=3;
                        plot_opt...);
            end
        end;

        ##
        ## Put it all together
        ##
        lay = @layout [a b c [d; e] f]

        plot_final = plot(
                plot_raw, plot_1d, #plot_ctr_true,
                plot_residual1, plot_ctr_guess, plot_ctr_true, plot_reconstruction,
                layout=lay)
        plot!(size=(2000, 250))

        return plot_final
end
