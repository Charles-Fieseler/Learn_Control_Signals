using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0f0, 50.0f0)

"""
Simple model of a spiking neuron, from the paper:
    Izhikevich, Eugene M. "Simple model of spiking neurons."
    IEEE Transactions on neural networks 14, no. 6 (2003): 1569-1572.
"""
function neuron_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    a, b = p
    v, u = u

    Ft = U_func_time(t)
    Fs = U_func_space(u)

    du[1] = 0.04*(v^2) + 5*v + 140 - u
    du[2] = a*(b*v - u)
    du .+= Ft .+ Fs
end


# This system also has a callback function
#   Default values define a "Regular Spiking" neuron
condition(u, t, integrator) = u[1] - 30
function affect!(integrator; c=-65, d=8)
    integrator.u[1] = c     # Reset
    integrator.u[2] += d    # Jump
end
neuron_reset = ContinuousCallback(condition, affect!,
                    save_positions = (false,false))

# Generate data, with a time component
p = [0.02, 0.2]
u0 = [-65.0, 0.0]
ts = range(tspan[1], tspan[2], length=5000)

function solve_neuron_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->neuron_system(du, u, p, t,
                                U_func_time=U_func_time,
                                U_func_space=U_func_space),
                    u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts,
                callback=neuron_reset);
    return sol
end

#####
##### Turing function for Bayesian parameter estimation
#####

# Unforced version for use with Turing.jl
function neuron_system(u, p, t)
    u = convert.(eltype(p),u)

    a, b = p
    v, u = u
    du = [0.04(v^2) + 5v + 140 - u; a*(b*v - u)]
    return du
end

# @model lorenz_grad_residual(y, dat) = begin
#     # Lorenz parameters
#     ρ ~ Normal(10, 10.0)
#     σ ~ Normal(10, 1.0)
#     β ~ Normal(5, 1.0)
#     params = [ρ, σ, β]
#     noise ~ Truncated(Normal(5, 5.0), 0, 20)
#
#     t = [0] # Not time dependent
#     for i in 1:size(y,2)
#         u = dat[:,i]
#         du_params = lorenz_system(u, params, t)
#         y[:, i] ~ MvNormal(du_params, [noise, noise, noise])
#     end
# end;


#####
##### True model in SINDy syntax
#####
# r, s, b = p
# #      x  y z  c xx xy xz yy yz zz
#  A = [[-s s 0  0  0  0  0  0  0  0];
#       [r -1 0  0  0  0 -1  0  0  0];
#       [0 0 -b  0  0  1  0  0  0  0]]
# n = size(A, 1)
# sindy_library = Dict(
#     "cross_terms"=>2,
#     "constant"=>nothing
# );
# core_dyn_true = sindyc_model(A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
#                             convert_string2function(sindy_library),
#                             ["x", "y", "z"])

# Actually export
export ts, solve_neuron_system, neuron_system
    # , core_dyn_true,
    # lorenz_grad_residual
