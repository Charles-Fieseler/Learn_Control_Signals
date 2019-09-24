using Statistics

include("sindyc.jl")
include("../utils/regression_utils.jl")

"""
Initialize the control signal using the residual of a linear fit model, dmd.
By default, removes small entries from the residual that may be noise

Normal use will be to run this to get the initial guess, then use 'sra_dmdc()'
to further refine the guess.
"""
function sra_dmd(dat, dat_grad;
                 quantile_threshold=0.95, add_const=false)
    n, m = size(dat)
    #   Note: add a constant field
    if add_const
        dat_and_ctr = vcat(dat, ctr);
    else
        dat_and_ctr = dat;
    end
    AB_guess = dat_grad / dat_and_ctr
    A_guess = AB_guess[:, 1:n]
    B_guess = AB_guess[:, n+1:end]
    # Stay in gradient space
    U_guess = calculate_residual(dat_and_ctr, dat_grad, AB_guess)
    # grad_guess = AB_guess*dat_and_ctr;
    # U_guess = dat_grad .- grad_guess
    # if using_previous_ctr
    #     U_guess = vcat(U_guess, ctr)
    # end
    # plot(U_func.(ts), label="Control Signal")
    # plot!(U_guess[2,:], label="Control signal guess")

    # Remove the smaller elements of U_guess
    sparsify_signal!(U_guess, quantile_threshold=quantile_threshold)

    return (U=U_guess, A=A_guess, B=B_guess)
end


"""
Initialize the control signal using the residual of a NONLINEAR fit model, SINDy.
By default, removes small entries from the residual that may be noise

Normal use will be to run this to get the initial guess, then use 'sra_sindy()'
to further refine the guess.
"""
function sra_sindy(dat, dat_grad;
                 quantile_threshold=0.95, library=nothing)
    n, m = size(dat)
    model = sindyc(dat, dat_grad,
                   library=library,
                   use_lasso=true)
    # Stay in gradient space
    U_guess = calculate_residual(model, dat, dat_grad)

    # Remove the smaller elements of U_guess
    sparsify_signal!(U_guess, quantile_threshold=quantile_threshold)

    return (U_guess=U_guess, model=model)
end


"""
Calculates the residual of the learned DMD dynamics
"""
function calculate_residual(dat, dat_grad, A)
    return dat_grad .- A*dat
end

"""
Calculates the residual of the learned SINDy dynamics
"""
function calculate_residual(m::sindyc_model, dat, dat_grad)
    # return dat_grad .- m(dat)
    return dat_grad .- intrinsic_dynamics(m, dat)
end


export sra_dmd, sra_sindy, calculate_residual, sparsify_signal!
