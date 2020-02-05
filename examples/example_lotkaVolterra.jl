using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0f0, 50.0f0)

"""
Lotka-volterra model from:
    https://en.wikipedia.org/wiki/Lotka–Volterra_equations
"""
function lv_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    Ft = U_func_time(t)
    Fs = U_func_space(u)
    α, β, δ, γ = p
    x, y = u

    xy = x*y
    du[1] = α*x - β*xy
    du[2] = δ*xy - γ*y

    du .+= Ft .+ Fs
end


# Generate data, with a time component
p = [2//3, 4//3, 1, 1]
u0 = [0.9, 0.9]
ts = range(tspan[1], tspan[2], length=5001)

function solve_lv_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->lv_system(du, u, p, t,
                                U_func_time=U_func_time,
                                U_func_space=U_func_space),
                    u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts);
    return sol
end

#####
##### Turing function for Bayesian parameter estimation
#####
# Unforced version for use with Turing.jl
function lv_system(u, p, t)
    u = convert.(eltype(p),u)
    α, β, δ, γ = p
    x, y = u

    xy = x*y
    du = [α*x - β*xy; δ*xy - γ*y]
    return du
end

#####
##### True model in SINDy syntax
#####
α, β, δ, γ = p
#      x   y   c   xx xy yy
 A = [[α   0   0   0 -β  0 ];
      [0  -γ   0   0  δ  0 ]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>2,
    "constant"=>nothing
);
core_dyn_true = sindyc_model(ts, A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["x", "y"])

# Actually export
export ts, solve_lv_system, lv_system, core_dyn_true
