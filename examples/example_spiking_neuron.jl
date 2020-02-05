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
    Ft = U_func_time(t)
    Fs = U_func_space(u)
    a, b = p
    v, u_var = u

    du[1] = 0.04*(v^2) + 5*v + 140 - u_var
    du[2] = a*(b*v - u_var)
    # TODO: Why is this breaking?
    # du .+= Ft .+ Fs
    tmp = Ft .+ Fs
    du[1] += tmp[1]
    du[2] += tmp[2]
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

#####
##### True model in SINDy syntax
#####
a, b = p
#      x    y  c   xx    xy yy
 A = [[5   -1  140  0.04  0  0];
      [a*b -a  0    0     0  0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>2,
    "constant"=>nothing
);
core_dyn_true = sindyc_model(ts, A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["x", "y"])

# Actually export
export ts, solve_neuron_system, neuron_system, core_dyn_true
    # ,lorenz_grad_residual
