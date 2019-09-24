
function nchoosek(n, k; with_replacement=true)
    perms = []
    all_branches = collect(1:n)
    for i in 2:k
        next_branches = collect(1:n)
        if !with_replacement
            deleteat!(next_branches, i)
        end
        all_branches = add_branches(all_branches, next_branches)
    end
    return all_branches
end

function add_branches(roots, options)
    new_branches = []
    for r in roots
        for o in options
            push!(new_branches, [r o])
        end
    end
    return new_branches
end

function calc_permutations(n, k)
    perms = []
    current_branches = collect(1:n)
    all_branches = nothing
    for i in 2:k
        tmp = copy(current_branches)
        all_branches = nothing
        for b in tmp
            next_branches = collect(b[end]:n)
            b = add_branches([b], next_branches)
            if all_branches == nothing
                all_branches = b
            else
                all_branches = vcat(all_branches, b)
            end
        end
        current_branches = all_branches
    end
    return all_branches
end

export nchoosek, calc_permutations
