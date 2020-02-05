using OrdinaryDiffEq, DifferentialEquations, Turing

tspan = (0.0f0, 50.0f0)

"""
Lorenz system with chaotic parameter values
"""
function lorenz_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    ρ, σ, β = p
    x, y, z = u

    Ft = U_func_time(t, u)
    Fs = U_func_space(u)

    du[1] = σ*(y-x)
    du[2] = x*(ρ-z) - y
    du[3] = x*y - β*z
    du .+= Ft .+ Fs
end

# Unforced version for use with Turing.jl
function lorenz_system(u, p, t)
    u = convert.(eltype(p),u)

    ρ, σ, β = p
    x, y, z = u
    du = [σ*(y-x); x*(ρ-z) - y; x*y - β*z]
    return du
end

# Generate data, with a time component
p = [28, 10, 8//3]
u0 = [10.0f0, 0.0f0, 0.0f0]
ts = range(tspan[1], tspan[2], length=5000)

function solve_lorenz_system(U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->lorenz_system(du, u, p, t,
                                U_func_time=U_func_time,
                                U_func_space=U_func_space),
                    u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts);
    return sol
end

#####
##### Turing function for Bayesian parameter estimation
#####
function lorenz_system(u, p, t)
    u = convert.(eltype(p),u)

    ρ, σ, β = p
    x, y, z = u
    du = [σ*(y-x); x*(ρ-z) - y; x*y - β*z]
    return du
end

@model lorenz_grad_residual(y, dat) = begin
    # Lorenz parameters
    ρ ~ Normal(10, 10.0)
    σ ~ Normal(10, 1.0)
    β ~ Normal(5, 1.0)
    params = [ρ, σ, β]
    noise ~ Truncated(Normal(5, 5.0), 0, 20)

    t = [0] # Not time dependent
    for i in 1:size(y,2)
        u = dat[:,i]
        du_params = lorenz_system(u, params, t)
        y[:, i] ~ MvNormal(du_params, [noise, noise, noise])
    end
end;


#####
##### True model in SINDy syntax
#####
r, s, b = p
#      x  y z  c xx xy xz yy yz zz
 A = [[-s s 0  0  0  0  0  0  0  0];
      [r -1 0  0  0  0 -1  0  0  0];
      [0 0 -b  0  0  1  0  0  0  0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>2,
    "constant"=>nothing
);
core_dyn_true = sindyc_model(ts, A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["x", "y", "z"])

# Actually export
export ts, solve_lorenz_system, lorenz_system, core_dyn_true,
    lorenz_grad_residual
