using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0f0, 50.0f0)

"""
Van der Pol oscillator model from wikipedia
"""
function vdp_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    Ft = U_func_time(t)
    Fs = U_func_space(u)
    μ = p[1]
    x, y = u

    du[1] = μ*(x - (x^3)/3 - y)
    du[2] = x/μ

    # Add controllers in
    du .+= Ft .+ Fs
end


# Generate data, with a time component
p = [1.0]
u0 = [1.0, 1.0]
ts = range(tspan[1], tspan[2], length=5001)

function solve_vdp_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->vdp_system(du, u, p, t,
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
# function vdp_system(u, p, t)
#     u = convert.(eltype(p),u)
#     beta, gamma = p
#     S, I, R = u
#
#     N = sum([S,I,R])
#     tmp = beta*I*S/N
#     du = [-tmp; tmp - gamma*I; gamma*I]
#     return du
# end

#####
##### True model in SINDy syntax
#####
μ = p
#      x  y  c  xx xy yy xxx  xxy xyy yyy
 A = [[μ -μ  0  0  0  0  -μ/3 0   0   0];
      [1/μ 0 0  0  0  0  0    0   0   0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>[2,3],
    "constant"=>nothing
);
core_dyn_true = sindyModel(ts, A,
                            convert_string2function(sindy_library),
                            ["x", "y"])

# Actually export
export ts, solve_vdp_system, vdp_system, core_dyn_true
