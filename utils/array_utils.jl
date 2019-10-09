
"""
Maps the function onto each row of the passed array
"""
function map_rows(f::Function, a::Array)
    n = size(a,1)
    vals = zeros(n)
    for i in 1:n
        vals[i] = f(a[i,:])
    end
    return vals
end

"""
Returns a function that applies the function-generating function 'f' to
    the rows of 'a', which can then be called with a SCALAR argument
"""
function generate_map_rows_function(f::Function, a::Array)
    n = size(a,1)
    func_list = []
    for i in 1:n
        push!(func_list, f(a[i,:]))
    end
    function F(t)
        return t .|> func_list
    end
    return F
end

"""
Drops dimensions that are equal to one in multidimensional arrays
    from: https://stackoverflow.com/questions/46289554/drop-julia-array-dimensions-of-length-1
"""
function drop_all_1dims(a)
    return dropdims(a, dims=Tuple(findall(size(a).==1)))
end


"""
Get contiguous true values, returning all start and stop times as well as
    the list of block lengths.
"""
function calc_contiguous_blocks(dat::Vector;
                                minimum_length=1)
    dat = Bool.(dat)
    all_starts = []
    all_ends = []
    current_length = 0
    for i in 1:length(dat)
        if dat[i]
            if current_length == 0
                push!(all_starts, i)
            end
            current_length += 1
        else
            if current_length > 0
                push!(all_ends, i-1)
                current_length = 0
            end
        end
    end
    if current_length > 0
        # Ended on 'true'
        push!(all_ends, length(dat))
    end

    block_lengths = all_ends .- all_starts .+ 1
    # Remove short ones
    long_enough = (block_lengths .>= minimum_length)
    all_starts = all_starts[long_enough]
    all_ends = all_ends[long_enough]
    block_lengths = block_lengths[long_enough]

    # Get the "good" indices
    true_indices = []
    for (s, e) in zip(all_starts, all_ends)
        true_indices = vcat(true_indices, s:e)
    end

    return (true_indices=true_indices,
            block_lengths=block_lengths,
            all_starts=all_starts,
            all_ends=all_ends)
end



export map_rows, generate_map_rows_function, drop_all_1dims,
        calc_contiguous_blocks
