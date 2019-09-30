
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


export map_rows, generate_map_rows_function, drop_all_1dims
