using PkgSRA
using Plots, Random, Flux, DiffEqFlux, OrdinaryDiffEq, Statistics
using DiffEqSensitivity
using ForwardDiff
using Interpolations
pyplot()
Random.seed!(11)

#####
##### Define the controlled dynamical system
#####
# Load example problem
include("paper_settings.jl")
include(EXAMPLE_FOLDERNAME*"example_lorenz.jl")

# Define the forcing functions
num_ctr = 5;
    U_starts = rand(num_ctr,1) .* tspan[2]
    U_width = 0.05;
    my_U_func_time(t, u) = U_func_time(t, u, U_width, U_starts, F_dim=3)

#####
##### Produce data
#####
sol = solve_lorenz_system(my_U_func_time)
dat = Array(sol)

# # Interpolated function of data
# dat_interp_vec = [
#     CubicSplineInterpolation(ts, Float64.(dat[i,:])) for i in 1:size(dat,1)
# ]
# function dat_interp_func(t)
#     return t .|> dat_interp_vec
# end

# Plot to make sure this works
# dat_interp = hcat(dat_interp_func.(collect(ts))...)
# plot(vec(dat_interp[1,:]), label="interp")
#     plot!(vec(dat[1,:]), label="dat")

numerical_grad = numerical_derivative(dat, ts)
diffeq_grad = hcat(sol(ts, Val{1})...)
true_grad = zeros(size(dat))
for i in 1:size(dat,2)
    true_grad[:,i] = lorenz_system(true_grad[:,i], dat[:,i], p, [0])
end
plot(numerical_grad[3,:], label="numerical")
    plot!(diffeq_grad[3,:], label="diffeq")
    plot!(true_grad[3,:], label="true uncontrolled")

U_true = zeros(size(dat))
for (i, t) in enumerate(ts)
    U_true[:,i] = my_U_func_time(t, dat[:,i])
end

#####
##### Define the sensitivity problem
#####
# Test to get the sensitivity to work!
# u0, tspan = Float64.(u0), Float64.(tspan)
# prob = ODELocalSensitivityProblem(lorenz_system,
#                                   u0, tspan, p)
# sol = solve(prob, Tsit5(), saveat=ts)

# Real problem: solve for the residual
# u0, tspan = Float64.(u0), Float64.(tspan)
#
# residual(du, u, p, t) = lorenz_system(du, u, p, t) .- dat_interp_func(t)
# prob = ODELocalSensitivityProblem(lorenz_system,
#                                   u0, tspan, p)
# sol = solve(prob, Tsit5(), saveat=ts)

# UPDATE: Just directly do ForwardDiff
# using Calculus
# function Lorenz_residual(p, t)
#   prob = ODEProblem(lorenz_system, eltype(p).(u0), eltype(p).((0, t)), p)
#   sol = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14, save_everystep=false)[end]
#   return sum(abs2, Array(sol) .- dat[:, i])
#   # return eltype(p).Array(sol) .- eltype(p).(dat[:, i])
# end
#
# # p = [1.5,1.0,3.0]
# i = 10
# fd_res = ForwardDiff.jacobian((p)->Lorenz_residual(p,i),Float64.(p))
# calc_res = Calculus.finite_difference_jacobian(
            # (p)->Lorenz_residual(p,i), Float64.(p))
# THE SECTION ABOVE WORKS (but not for the sum(abs))

# Match to the CONTROLLED solution
# dg(out,u,i) = (out.=sol[i].-u)
# res = adjoint_sensitivities(sol,Tsit5(),dg,ts)#,abstol=1e-14,
                            # reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

# dg(out,u,i) = (out.=sol[i].-u)
# dg(out,u,p,t,i) = (out.=sol[i].-u)
# res = adjoint_sensitivities(sol,Vern9(),dg,ts,abstol=1e-14,
                            # reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

using ForwardDiff,Calculus
prob = lorenz_system
function G(p, t)
    tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
    sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=t)
    grad = sol(t, Val{1})
    u = sol(t)
    return grad .- my_U_func_time(t, u)
    # A = convert(Array,sol)
    # sum(((1-A).^2)./2)
end
G(p, 10)
res2 = ForwardDiff.gradient((p) -> G,p)
