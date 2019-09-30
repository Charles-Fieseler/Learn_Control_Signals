using Turing, Distributions

"""
Generate test trajectories from the posterior
"""
function sample_posterior_grad(chain, dat, sample_ind, vars, system_model;
                                num_samples=100, t = [0])
    param_samples = sample(chain, num_samples)
    all_vals = zeros(num_samples, length(sample_ind), size(dat,1))
    all_noise = zeros(num_samples)
    for i in 1:num_samples
        these_params = [param_samples[v].value[i] for v in vars]
        all_noise[i] = param_samples[:noise].value[i]
        for (i_save, i_dat) in enumerate(sample_ind)
            all_vals[i, i_save, :] = system_model(dat[:,i_dat],
                                    these_params, t)
        end
    end

    return (sample_trajectories=all_vals, sample_noise=all_noise)
end


"""
Generate a chain with settings that have worked well for me
    Uses NUTS sampler
"""
function generate_chain(dat, numerical_grad, turing_model;
                        iterations=1000,
                        num_training_pts=500, start_ind=1)
    n_adapts = Int(iterations/5)
    j_max = 1.0
    # Try to predict the GRADIENT from data
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
    chain = sample(turing_model(y, train_ind),
                    NUTS(iterations, n_adapts, 0.6j_max));
    return (chain=chain, train_ind=train_ind)
end
