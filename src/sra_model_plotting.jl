# Plotting the data structure for the state of an SRA analysis
#   See also: sra_model_object.jl for the saved states


#####
##### Basic data plotting
#####
function plot_data(m::sra_stateful_object)
    names = m.sra_parameters
    plot(m.ts,m.dat[1,:], label=names[1]);
    for i = 1:size(dat,1)
        plot!(m.ts,m.dat[i,:], label=names[i]);
    end
end

function plot_data(m::sra_stateful_object, which_dim=1)
    names = m.sra_parameters
    plot(m.ts,m.dat[which_dim,:], label=names[which_dim]);
end

#####
##### Plotting against the truth
#####
function plot_true_grads(m::sra_stateful_object,
            t::sra_truth_object,
            which_dim=1)
    plot(m.ts, t.true_grad[which_dim,:]);
    plot!(m.ts, m.numerical_grad[which_dim,:])
end

function plot_data_and_control(m::sra_stateful_object,
            t::sra_truth_object)
    plot_data(m)
    plot!(m.ts, t.U_true)
end

#####
##### Sindy equation printing
#####
function print_current_equations(m::sra_stateful_object, kwargs...)
    println("Current model equations:")
    print_equations(m.sindy_model, kwargs...)
end

function print_true_equations(t::sra_truth_object, kwargs...)
    println("True equations:")
    print_equations(t.core_dyn_true, kwargs...)
end


#####
##### Sindy equation simulations
#####
function plot_sindy_model(m::sra_stateful_object, which_dim=1)
    plot_data(m, which_dim)
    plot!(m.sindy_dat[which_dim,:], label=m.variable_names[which_dim]);
    title!("Current model")
end
