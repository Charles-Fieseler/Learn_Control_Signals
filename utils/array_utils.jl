
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


export map_rows
