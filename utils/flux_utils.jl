using Flux

"""
Helper function to train a NN in a loop with a maximum number of iterations.
    Breaks early if a stall is detected
    Returns a boolean for convergence to the given tolerance
"""
function train_in_loop(ps, tol, loss;
                rate=1e-3, stall_tol=nothing,
                dat_size=50, max_iter=100,
                plot_func=nothing, plot_truth=nothing)
    if stall_tol == nothing
        stall_tol = tol / 100
    end
    dat_dummy = Iterators.repeated((), dat_size)
    opt = ADAM(rate)
    iter = 1
    current_loss = loss() + 100 # So it doesn't stop early
    is_converged = true
    while current_loss > tol
        prev_loss = current_loss
        current_loss = loss()
        if prev_loss - current_loss < stall_tol
            println("Stall detected at iteration $iter; aborting")
            is_converged = false
            break
        end

        Flux.train!(loss, ps, dat_dummy, opt, cb = ()->@show loss())
        iter += 1

        if plot_func !== nothing
            pt = plot(plot_func(), label="NN")
            if plot_truth !== nothing
                plot!(plot_truth, label="True control")
            end
            title!("Iteration $iter")
            display(pt)
        end
        if iter > max_iter
            println("Reached maximum iterations; aborting")
            is_converged = false
            break
        end
    end

    return is_converged
end


export train_in_loop
