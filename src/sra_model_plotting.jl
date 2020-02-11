# Plotting the data structure for the state of an SRA analysis
#   See also: sra_model_object.jl for the saved states
#   See also: sra_model_function.jl for the analysis steps

import PkgSRA.plot_sindy_model

#####
##### Basic data plotting
#####
# function plot_data(m::sra_stateful_object)
#     names = m.sra_parameters
#     plot(m.ts,m.dat[1,:], label=names[1]);
#     for i = 2:size(m.dat,1)
#         plot!(m.ts,m.dat[i,:], label=names[i]);
#     end
#     title!("Raw data")
# end

function plot_data(m::sra_stateful_object, which_dim=1)
    names = m.sindy_model.variable_names
    plot(m.ts,m.dat[which_dim,:], label=names[which_dim]);
    title!("Raw data (variable $which_dim)")
end

function plot_derivative(m::sra_stateful_object, which_dim=1)
    names = m.sindy_model.variable_names
    plot(m.ts,m.numerical_grad[which_dim,:], label=names[which_dim]);
    title!("Numerical derivative (variable $which_dim)")
end

#####
##### Plotting against the truth
#####
function plot_true_grads(m::sra_stateful_object,
            t::sra_truth_object,
            which_dim=1)
    plot(m.ts, t.true_grad[which_dim,:], label="True gradient");
    plot!(m.ts, m.numerical_grad[which_dim,:], label="Numerical gradient")
end

# function plot_data_and_control(m::sra_stateful_object,
#             t::sra_truth_object)
#     plot_data(m)
#     plot!(m.ts, t.U_true, label="True control")
#     title!("Data and controller")
# end

function plot_data_and_control(m::sra_stateful_object,
            t::sra_truth_object; which_dim=1)
    plot_data(m, which_dim)
    plot!(m.ts, t.U_true[which_dim,:], label="True control")
    title!("Data and controller (Variable $which_dim)")
end

function plot_subsampled_points_and_control(m::sra_stateful_object,
            t::sra_truth_object, which_dim=1)
    plot_data_and_control(m, t, which_dim=which_dim)
    dat_sub = m.dat[:, m.subsample_ind]
    scatter!(m.ts[m.subsample_ind], dat_sub[which_dim,:], color=:blue)
    title!("Subsampled points plotted on data (Variable $which_dim)")
end

#####
##### Sindy equation printing
#####
function print_current_equations(m::sra_stateful_object; kwargs...)
    i = m.i
    if i > 1
        println("Current model equations for iteration $i:")
    else
        println("Naive model equations (initial iteration):")
    end
    print_equations(m.sindy_model; kwargs...)
end

function print_true_equations(t::sra_truth_object; kwargs...)
    println("True equations:")
    print_equations(t.core_dyn_true; kwargs...)
end


#####
##### Sindy equation simulations
#####
function plot_sindy_model(m::sra_stateful_object, which_dim=1)
    if m.sindy_dat == nothing
        simulate_model(m)
    end
    plot_data(m, which_dim)
    name = m.sindy_model.variable_names[which_dim]
    plot!(m.ts, m.sindy_dat[which_dim,:],
        label="Simulation ($name)");
    title!("Integration of current model")
end


function plot_sindy_derivatives(m::sra_stateful_object, which_dim=1)
    plot_derivative(m, which_dim)
    name = m.sindy_model.variable_names[which_dim]
    sindy_deriv = m.sindy_model(m.dat, 0)
    plot!(m.ts, sindy_deriv[which_dim,:],
        label="Simulation ($name)");
    title!("Derivatives of current model")
end

#####
##### Subsampling visualizations
#####
function plot_subsampled_points(m::sra_stateful_object, which_dim=1)
    plot_data(m, which_dim)
    dat_sub = m.dat[:, m.subsample_ind]
    scatter!(m.ts[m.subsample_ind], dat_sub[which_dim,:], color=:blue)
    title!("Subsampled points plotted on data (Variable $which_dim)")
end

function plot_residual(m::sra_stateful_object, which_dim=1)
    # TODO: This has to recalculate the residual
    residual = calc_residual(m)

    plot(residual[which_dim,:], label="Residual")
    hline!([m.noise_guess], label="Noise line");
    hline!([-m.noise_guess], label="Noise line")
end

"""
Plots the data and the integrated model.
"""
function plot_subsampled_simulation(m::sra_stateful_object, which_dim=1)
    plot_sindy_model(m, which_dim)
    dat_sub = m.dat[:, m.subsample_ind]
    ts_sub = m.ts[m.subsample_ind]
    scatter!(ts_sub, dat_sub[which_dim,:], color=:blue)
    xlims!((ts_sub[1], ts_sub[end]))
    title!("Subsampled points with integrated model (Variable $which_dim)")
end

"""
Plots the numerical derivative and the derivatives evaluated from the model.
    Note: no integration is performed
"""
function plot_subsampled_derivatives(m::sra_stateful_object, which_dim=1)
    plot_sindy_derivatives(m, which_dim)
    deriv_sub = m.numerical_grad[:, m.subsample_ind]
    ts_sub = m.ts[m.subsample_ind]
    scatter!(ts_sub, deriv_sub[which_dim,:], color=:blue)
    xlims!((ts_sub[1], ts_sub[end]))
    title!("Subsampled points with sindy derivatives (Variable $which_dim)")
end

##### Export
export plot_data, plot_data_and_control, plot_subsampled_points,
    plot_sindy_model, print_current_equations, print_true_equations,
    plot_subsampled_derivatives, plot_residual,
    plot_subsampled_points_and_control, plot_subsampled_simulation
