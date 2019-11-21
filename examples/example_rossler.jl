using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0, 200.0)

"""
Rossler system with chaotic parameter values
"""
function rossler_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    a, b, c = p
    x, y, z = u

    Ft = U_func_time(t)
    Fs = U_func_space(u)

    du[1] = -y - z
    du[2] = x + a*y
    du[3] = b + x*z - c*z
    du .+= Ft .+ Fs
end

# Generate data
p = [0.1, 0.1, 14]
u0 = [1.0, 1.0, 0.0]
ts = range(tspan[1], tspan[2], length=5000)

function solve_rossler_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->rossler_system(du, u, p, t,
                                U_func_time=U_func_time,
                                U_func_space=U_func_space),
                    u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts);
    return Array(sol)
end

# True model in SINDy syntax
#     x y z c xx xy xz yy yz zz
 A = [[0 -1 -1 0   0 0 0 0 0 0];
      [1 0.1 0 0   0 0 0 0 0 0];
      [0 0 -14 0.1 0 0 1 0 0 0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>2,
    "constant"=>nothing
);
core_dyn_true = sindyc_model(A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["x", "y", "z"])

# Actually export
export ts, solve_rossler_system, rossler_system, core_dyn_true
