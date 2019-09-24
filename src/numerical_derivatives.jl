
using Interpolations

"""
Does a cubic spline interpolation and then takes a derivative across
the row for each column
"""
function numerical_derivative(dat, ts)
    # Take the approximate derivative in the row direction
    interp = []
    dat_grad = zeros(size(dat))
    if ts isa Array
        # Breaks the interpolation function if 'ts' is an array
        dt = ts[2]-ts[1]
        ts = ts[1]:dt:ts[end]
    end

    for i in 1:size(dat,1)
        push!(interp,CubicSplineInterpolation(ts, dat[i,:]))
        for (i2, t) in enumerate(ts)
            dat_grad[i,i2] = Array(Interpolations.gradient(interp[i], t))[1]
        end
    end
    return dat_grad
end

export numerical_derivative
