using OrdinaryDiffEq, DifferentialEquations
using Flux

tspan = (0.0f0, 10.0f0)

"""
Simple ball in one spatial dimension, for 3 dimensions total:
    x, v, a
"""
function ball_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    g = p
    Ft = U_func_time(t, u)
    Fs = U_func_space(u)

    du[1] = u[2]
    du[2] = -g
    du .+= Ft .+ Fs
end

"""
Similar to the falling ball system, but with deformations recorded in a third
variable. Comes with nonlinearities, and requires an extra parameter to set
the strength

In words, the three variables now are:
    Position; normal
    Velocity; affected by gravity, increases (to 0) if close to the ground,
            and increases again as deformation decreases back to 0
    Deformation; 0 until near the ground, where it deforms at a certain rate
            and then reforms at an ideally slower rate
"""
function deformed_ball_system(du, u, p, t;
                     U_func_time=U_func_time_trivial,
                     U_func_space=U_func_space_trivial)
    # g, deformation_rate, restoration_rate = p
    g, r, k, damping = p

    Ft = U_func_time(t)
    Fs = U_func_space(u)

    # May allow the ball to go slightly negative
    # r = 0.1; # Radius of ball
    Δ = max(0.0, r - u[1])
    damping_force = Δ>0.0 ? -damping*u[2] : 0.0

    du[1] = u[2] + Fs[1]
    du[2] = -g + k*Δ + damping_force + Fs[2] + Ft

    # ground_closeness = deformation_rate*σ(100.0*(0.01-u[1]));
    # du3 = ground_closeness*(-u[2]) - restoration_rate*u[3] + Fs[3]

    # du[1] = u[2] + Fs[1]
    # du[2] = -g - du3 + Fs[2] + Ft
    # du[3] = du3
    # du[2] = -g + restoration_rate*u[3] + Fs[2] + Ft
    # du[3] = ground_closeness - restoration_rate*u[3] + Fs[3]
end

# Parameters
p1 = 9.81
u1 = [10.0f0, 0.0f0]

# p2 = [9.81, 1000.0, 100.0]
p2 = [9.81, 0.1, 10000.0, 5.0]
u2 = [1.0f0, 0.0f0]

ts = range(tspan[1], tspan[2], length=1000)

function solve_ball_system(;U_func_time = U_func_time_trivial,
                           U_func_space = U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->
                ball_system(du, u, p, t,
                            U_func_time=U_func_time,
                            U_func_space=U_func_space),
                    u1, tspan, p1)
    sol = solve(prob, Tsit5(), saveat=ts);
    return sol
end

function solve_deformed_ball_system(U_func_time = U_func_time_trivial,
                                    U_func_space = U_func_space_trivial)
    prob = ODEProblem((du, u, p, t)->
                deformed_ball_system(du, u, p, t,
                            U_func_time=U_func_time,
                            U_func_space=U_func_space),
                    u2, tspan, p2)
    sol = solve(prob, Tsit5(), saveat=ts);
    return Array(sol)
end

# Also define true dynamics in NN syntax
core_dyn_true = Dense(Float32.([0.0 1.0; 0.0 0.0]),
                      Float32.([0.0;-p1]),
                      identity)

export solve_ball_system, ball_system,
        solve_deformed_ball_system, deformed_ball_system,
        core_dyn_true

## Old scratch
# Try to define using a jump system
# bounce_rate(u,p,t) = u[1]<0.0 ? 1.0 : 0.0
# bounce_effect!(integrator) = integrator.u[2] += 10.0
# bounce_jump = VariableRateJump(bounce_rate, bounce_effect!)
#
# prob = ODEProblem(ball_system, u0, tspan, p)
# jump_problem = JumpProblem(prob, Direct(), bounce_jump)
# sol = solve(jump_problem, Tsit5())


# SCRATCH: Bounce using callbacks
# From: http://docs.juliadiffeq.org/latest/features/callback_functions.html#Example-1:-Bouncing-Ball-1
# function condition(u,t,integrator) # Event when event_f(u,t) == 0
#   u[1]
# end
# function affect!(integrator) # The actual bounce
#   integrator.u[2] = -integrator.u[2]
# end
# # Note: by default the callback will cause extra saves
# cb = ContinuousCallback(condition,affect!,
#                         save_positions = (false,false))
# prob = ODEProblem(ball_system, u0, tspan, p)
# sol = solve(prob, Tsit5(), saveat=ts, callback=cb)
            #initialize_save=false)
# plot(sol)
