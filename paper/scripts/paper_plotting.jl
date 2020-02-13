
#####
##### Global settings
#####
plot_opt = Dict(:titlefontsize=>28,
        :xticks=>false, :yticks=>false, :zticks=>false,
        :legend=>false, :fontfamily=>:serif)

################################################################################
#####
##### Define a function for library examples
#####

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

################################################################################
#####
##### Define a function for noise studies
#####

function plot_library_noise(this_dat_name, system_name="";
                        plot_opt=plot_opt)

        plot_opt[:xticks] = true
        plot_opt[:yticks] = true
        plot_opt[:tickfontsize] = 16
        # plot_opt[:colorbar] = true
        plot_opt[:color] = :Spectral#:RdYlBu

        ## Load
        this_dat_name = DAT_FOLDERNAME*this_dat_name;

        fname = this_dat_name*"metadata.bson"
        @load fname noise_vals control_signal_vals coef_norm

        fname = this_dat_name*"coefficients.bson"
        @load fname all_naive_err all_err

        fname = this_dat_name*"derivatives.bson";
        @load fname all_err_deriv_subsample all_naive_err_deriv
        # fname = this_dat_name*"coefficients.bson"
        # @load fname noise_vals vec_naive std_naive vec_err std_err
        #
        # fname = this_dat_name*"derivatives.bson";
        # @load fname vec_deriv std_deriv vec_naive_deriv std_naive_deriv
        # vec_naive, std_naive = mean_and_std(all_naive_err./coef_norm, 2)
        # vec_err, std_err = mean_and_std(all_err./coef_norm, 2)
        # vec_deriv, std_deriv = mean_and_std(all_err_deriv_subsample, 2)
        # vec_naive_deriv, std_naive_deriv = mean_and_std(all_naive_err_deriv, 2)

        ## Process
        # heatmap_coef_naive = mean(all_naive_err./coef_norm, dims=3)[:,:,1]
        # heatmap_coef_final = mean(all_err./coef_norm, dims=3)[:,:,1]

        heatmap_deriv_naive = mean(all_naive_err_deriv, dims=3)[:,:,1]
        heatmap_deriv_final = mean(all_err_deriv_subsample, dims=3)[:,:,1]
        clims = (0, 2*median(heatmap_deriv_naive))
        plot_opt[:clims] = clims

        plot_deriv = heatmap(control_signal_vals, noise_vals,
                heatmap_deriv_final; plot_opt...)
            xlabel!("Percent Data Perturbed", guidefontsize=20)
            xticks!(control_signal_vals)
            ylabel!("Noise", guidefontsize=20)
            title!(system_name*" Final Error")

        plot_opt[:yticks] = false
        plot_opt[:colorbar] = false
        plot_improve = heatmap(control_signal_vals, noise_vals,
                heatmap_deriv_naive .- heatmap_deriv_final; plot_opt...)
            xlabel!("Percent Data Perturbed", guidefontsize=20)
            xticks!(control_signal_vals)
            # ylabel!("Noise")
            title!("Improvement over SINDy")

        ## First panel: Error in coefficients
        # coef_plot = plot(noise_vals, vec_naive, ribbon=std_naive,
        #                 label="Intial SINDy", lw=3; plot_opt...)
        #     plot!(noise_vals, vec_err, ribbon=std_err, lw=3,
        #                 label="Final iteration")
        #     xlabel!("Noise", guidefontsize=24)
        #     ylabel!("Error", guidefontsize=20)
        #     # title!("Fractional Error in Coefficients")
        #     title!(system_name*" (coefficients)")

        ## Second panel: Error in derivatives
        # deriv_plot = plot(noise_vals, vec_naive_deriv, ribbon=std_naive_deriv,
        #                 label="Intial SINDy", lw=3; plot_opt...)
        #     plot!(noise_vals, vec_deriv, ribbon=std_deriv, lw=3,
        #                 label="Final iteration")
        #     xlabel!("Noise", guidefontsize=24)
        #     ylabel!("Error", guidefontsize=20)
        #     # title!("Error in Model Derivatives")
        #     title!(system_name*" (derivatives)")

        ##
        ## Put it all together
        ##
        lay = @layout [a b]

        # plot_final = plot(coef_plot, deriv_plot, layout=lay)
        plot_final = plot(plot_deriv, plot_improve, layout=lay)
        plot!(size=(2000, 250))

        ## Save
        # savefig(plot_final, this_dat_name * ".png")

        return plot_final
end



##
export plot_library, plot_library_noise
