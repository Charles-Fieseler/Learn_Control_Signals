using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0f0, 50.0f0)

"""
FitzHugh Nagumo oscillator model from wikipedia
"""
function fhn_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    Ft = U_func_time(t)
    Fs = U_func_space(u)
    a, b, I_external = p[1]
    v, ω = u

    du[1] = v - (v^3)/3 - ω + I_external
    du[2] = v + a - b*ω

    # Add controllers in
    du .+= Ft .+ Fs
end


# Generate data, with a time component
p = [1.0, 0.7, 0.5]
u0 = [1.0, 1.0]
ts = range(tspan[1], tspan[2], length=5001)

function solve_fhn_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->fhn_system(du, u, p, t,
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
[a, b, I] = p
#      x  y  c  xx xy yy xxx  xxy xyy yyy
 A = [[1 -1  I  0  0  0  -1/3 0   0   0];
      [1 -b  a  0  0  0  0    0   0   0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>[2,3],
    "constant"=>nothing
);
core_dyn_true = sindyc_model(A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["v", "ω"])

# Actually export
export ts, solve_fhn_system, fhn_system, core_dyn_true
