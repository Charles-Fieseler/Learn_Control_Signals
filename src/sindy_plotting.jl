
# include("sindy.jl")
import PkgSRA.plot_sindy_model

# Extending other function
function plot_sindy_model(m::DynamicalSystemModel, u0;
                        which_dim=1)
    recon = simulate_model(m, u0)
    names = m.variable_names
    plot(m.ts, recon[which_dim,:], label=names[which_dim]);
    title!("Sindy reconstruction (variable $which_dim)")
end

export plot_sindy_model
