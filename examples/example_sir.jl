using OrdinaryDiffEq, DifferentialEquations

tspan = (0.0f0, 50.0f0)

"""
Simple SIR disease model from:
    https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology

Note: this does not include birth and death rates
"""
function sir_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    Ft = U_func_time(t)
    Fs = U_func_space(u)
    beta, gamma = p
    S, I, R = u

    N = sum([S,I,R])
    tmp = beta*I*S/N
    du[1] = -tmp
    du[2] = tmp - gamma*I
    du[3] = gamma*I

    # This forcing function must keep the overall population constant
    du[1] += -Ft[1]
    du[3] += Ft[1]
    # du .+= Ft .+ Fs
end


# Generate data, with a time component
p = [0.5, 0.2]
u0 = [999.0, 1.0, 0.0]
ts = range(tspan[1], tspan[2], length=5001)

function solve_sir_system(;U_func_time=U_func_time_trivial,
                              U_func_space=U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->sir_system(du, u, p, t,
                                U_func_time=U_func_time,
                                U_func_space=U_func_space),
                    u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=ts);#, adaptive=true,
        #dt=1e-5);#, abstol=1e-12, reltol=1e-5);
    # sol = solve(prob, AB4(), saveat=ts, dt=1e-5);
    return sol
end

#####
##### Turing function for Bayesian parameter estimation
#####
# Unforced version for use with Turing.jl
function sir_system(u, p, t)
    u = convert.(eltype(p),u)
    beta, gamma = p
    S, I, R = u

    N = sum([S,I,R])
    tmp = beta*I*S/N
    du = [-tmp; tmp - gamma*I; gamma*I]
    return du
end

#####
##### True model in SINDy syntax
#####
a, b = p
a = a / sum(u0)
#      S  I  R  c SS SI SR II IR RR
 A = [[0  0  0  0  0 -a  0  0  0  0];
      [0 -b  0  0  0  a  0  0  0  0];
      [0  b  0  0  0  0  0  0  0  0]]
n = size(A, 1)
sindy_library = Dict(
    "cross_terms"=>2,
    "constant"=>nothing
);
core_dyn_true = sindyc_model(ts, A, zeros(n,1), zeros(1, 1), (t)->zeros(1),
                            convert_string2function(sindy_library),
                            ["S", "I", "R"])

# Actually export
export ts, solve_sir_system, sir_system, core_dyn_true
    # ,lorenz_grad_residual
