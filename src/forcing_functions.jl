using Random
using Interpolations

U_func_time_trivial(t, u) = zeros(size(u));
U_func_space_trivial(u) = zeros(size(u));

#####
##### Spike-like controllers
#####
# FROM: https://discourse.julialang.org/t/diffeqflux-with-time-as-additional-input-to-neural-ode/26456/3
function U_func_time(t, u, U_width, U_starts;
                        F_dim=2, amplitude=1.0)
    F = zeros(size(u))
    if any(i -> i < U_width, abs.(U_starts .- t))
        F[F_dim] = amplitude
    end
    return F
end

function U_func_time_multivariate(t, u, U_widths, U_starts;
                                F_dim=[2], amplitudes=nothing)
    if amplitudes == nothing
        amplitudes = ones(size(F_dim))
    end
    F = zeros(size(u))
    # Each forced dimension can have different timing
    for i_dim in 1:length(F_dim)
        if any(i -> i < U_widths[i_dim], abs.(U_starts[i_dim,:] .- t))
            F[i_dim] = amplitudes[i_dim]
        end
    end
    return F
end

function spring_forcing_example(tspan)
    Random.seed!(11)
    num_ctr = 2;
    U_starts = rand(num_ctr,1) .* tspan[2]
    U_width = 0.05;
    U_func(t, u) = U_func_time(t, u, U_width, U_starts)

    return U_func
end


#####
##### Piecewise constant controllers
#####

# Returns the function with signature:
#   f = generate_U_func_piecewise(t, u)
#   f(t, u) # control at that time
function generate_U_func_piecewise(tspan, u_size;
                                    num_jumps = 10,
                                    U_min=-1, U_max=1,
                                    F_dim=2)
    jump_locations = sort(rand(num_jumps) .* tspan[end])
    jump_locations = [jump_locations; tspan[end]]
    U_series = zeros(u_size)
    my_rand() = (U_max-U_min)*rand() + U_min
    let jump_ind=1, current_val=my_rand()
        for i in 1:length(tspan)
            if tspan[i] > jump_locations[jump_ind]
                jump_ind += 1
                current_val = my_rand()
            end
            U_series[F_dim, i] = current_val
        end
    end
    # Generates a function that can be called like f(t)
    # return U_series
    U_func = generate_map_rows_function(
                (x)->CubicSplineInterpolation(tspan, x), U_series)
    return (t, tmp)->U_func(t) # Needs the right signature
end



#####
##### Coordinate (spatial) controllers
#####
function U_func_hard_wall(X, x_dim=1, v_dim=2, wall_strength=100.0)
    F = zeros(size(X))
    X[x_dim]<0.0 ? F[v_dim]=wall_strength : return F
    # F = [wall_strength*σ(-5.0(x+0.0)), 0.0, 0.0]
    return F
end

function U_func_soft_wall(X, x_dim=1, v_dim=2, wall_strength=100.0)
    F = zeros(size(X))
    F[y_dim] = wall_strength*σ(-10.0*X[x_dim])
    return F
end

function U_func_spring(X; k=1000.0, r=0.1, damping=1.0,
                       r_dim=1, F_dim=2, damp_dim=2)
    F = zeros(size(X))
    # Damped spring at the 0 coordinate
    if r - X[r_dim] > 0.0
        Δ = r - X[r_dim]
        F[F_dim] = k*Δ + -damping*abs(X[damp_dim])
    end
    return F
end

function U_func_spring_sphere(X; k=1000.0,
                              r=0.1, location=nothing, damping=1.0,
                              F_dim=2, damp_dim=2)
    F = zeros(size(X))
    if location == nothing
        # Default is the origin
        location = zeros(size(X))
    end
    Δ = sum(abs2,location .- X)
    if Δ < r
        F[F_dim] = k*Δ + -damping*abs(X[damp_dim])
    end
    return F
end


#####
##### Export
#####
export U_func_time, U_func_time_multivariate,
        spring_forcing_example, U_func_spring_sphere,
        U_func_hard_wall, U_func_soft_wall, U_func_spring,
        U_func_time_trivial, U_func_space_trivial,
        generate_U_func_piecewise
