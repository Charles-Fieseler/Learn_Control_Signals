using OrdinaryDiffEq, DifferentialEquations
using Flux

tspan = (0.0f0, 4.0f0)

"""
Damped mass-spring system, with parameters '[m, k, c]' passed as 'p'
"""
function msd_system(du, u, p, t;
                    U_func_time=U_func_time_trivial,
                    U_func_space=U_func_space_trivial)
    m, k, c = p  # Mass, spring, damper

    Ft = U_func_time(t, u);
    Fs = U_func_space(u);
    g = 9.81
    # Forced linear system
    du[1] = u[2] + Fs[1] + Ft[1] # x = ẋ
    du[2] = (g*m - k*u[1] - c*u[2])/m + Fs[2] + Ft[2]
end

# Generate data, with a time component
m = 1.
k = 5.
c = 1.
g = 9.81
p1 = [m, k, c]

# Also just generate the true dynamics in NN form, without F
core_dyn_true = Dense(Float32.([0. 1.; -k/m -c/m]),
                      Float32.([0;-g]),
                      identity)

u0 = [1.0f0, 0.0f0]
ts = range(0.0f0, 4.0f0, length=300)

function solve_msd_system(U_func_time = U_func_time_trivial,
                          U_func_space = U_func_space_trivial)
    # msd_system =
    prob = ODEProblem((du, u, p, t)->
                msd_system(du, u, p, t,
                           U_func_time=U_func_time,
                           U_func_space=U_func_space),
                    u0, tspan, p1)
    sol = solve(prob, Tsit5(), saveat=ts);
    return sol
end

# dat = solve_msd_system(U_func)

# Helper functions and plotting
# plot(U_func.(ts), label="Control Signal")
# plot(sol)

# Actually export
# export dat, ts, core_dyn_true, solve_msd_system
export ts, core_dyn_true, solve_msd_system
