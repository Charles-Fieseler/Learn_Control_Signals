using Turing, Distributions


#####
##### Sampling from posteriors
#####

"""
Generate test realizations of the gradient from the posterior of the system
    parameters. No integration is performed
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
Generate test realizations of the gradient from the posterior of the system
    parameters. No integration is performed
"""
function sample_posterior_trajectories(chain, dat, sample_ind, vars, system_model,
                                tspan, ts;
                                num_samples=100)
    param_samples = sample(chain, num_samples)
    all_vals = []
    all_noise = zeros(num_samples)
    for i in 1:num_samples
        these_params = [param_samples[v].value[i] for v in vars]
        all_noise[i] = param_samples[:noise].value[i]
        prob = ODEProblem(system_model, dat[:,sample_ind[1]], tspan, these_params)
        sol = solve(prob, Tsit5(), saveat=ts);
        push!(all_vals, Array(sol))
    end

    return (sample_trajectories=all_vals, sample_noise=all_noise)
end

#####
##### Generating the posteriors via NUTS
#####

"""
Generate a chain with settings that have worked well for me
    Uses NUTS sampler

Input:
    dat - Each row is a variable
    numerical_grad - Each row is a variable
    turing_model - a model which takes in the data and predicts the gradient
        e.g:
        turing_model(grad, data)
"""
function generate_chain(dat, numerical_grad, turing_model;
                        opt=NUTS,
                        iterations=1000,
                        num_training_pts=500, start_ind=1)
    n_adapts = Int(minimum([iterations/5, 100]))
    j_max = 1.0
    # Try to predict the GRADIENT from data
    train_ind = start_ind:num_training_pts+start_ind-1
    y = numerical_grad[:,train_ind]
    x = dat[:,train_ind]
    # Turing 0.6.* syntax
    # chain = sample(turing_model(y, x),
    #                 NUTS(iterations, n_adapts, 0.6j_max));
    # Turing 0.7 syntax
    chain = sample(turing_model(y, x),
                    NUTS(n_adapts, 0.6j_max), iterations);
    return (chain=chain, train_ind=train_ind)
end


export sample_posterior_trajectories, sample_posterior_grad,
    generate_chain
