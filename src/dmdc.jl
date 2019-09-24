"""
Naive implementation of Dynamic Mode Decomposition with control (DMDc)

Defaults to predicting the second input as a function of the first.
    Can also be passed a 'mode', "discrete" or "continuous" to process the
    data 'X' to produce the variable to be solved for
"""
function dmdc(X, X_grad=nothing, U=nothing; mode="discrete")
    if X_grad == nothing
        if mode=="discrete"
            X_grad = X[:,2:end]
            X = X[:,1:end-1]
        elseif mode=="continuous"
            X_grad = numerical_derivative(X)
        end
    end

    U == nothing ? (Ω = X) : (Ω = vcat(X,U))
    # Solve for dynamics and separate out
    AB = X_grad / Ω
    if U == nothing
        A = AB
        B = nothing
    else
        n, m = size(X)
        A = AB[:, 1:n]
        B = AB[:, n+1:end]
    end

    return (A=A, B=B)
end

export dmdc
