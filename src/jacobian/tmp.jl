" https://github.com/JuliaDiff/SparseDiffTools.jl
" MIT License

"""
    color_graph(g::VSafeGraph, alg::GreedyD1Color)

Find a coloring of a given input graph such that
no two vertices connected by an edge have the same
color using greedy approach. The number of colors
used may be equal or greater than the chromatic
number `χ(G)` of the graph.
"""
function color_graph(g::VSafeGraph, alg::GreedyD1Color)
    v = nv(g)
    result = zeros(Int, v)
    result[1] = 1
    available = BitVector(undef, v)
    fill!(available, false)
    for i in 2:v
        for j in inneighbors(g, i)
            if result[j] != 0
                available[result[j]] = true
            end
        end
        for cr in 1:v
            if !available[cr]
                result[i] = cr
                break
            end
        end
        fill!(available, false)
    end
    return result
end


"""
    _cols_by_rows(rows_index,cols_index)

Returns a vector of rows where each row contains
a vector of its column indices.
"""
function _cols_by_rows(rows_index, cols_index)
    nrows = isempty(rows_index) ? 0 : maximum(rows_index)
    cols_by_rows = [eltype(rows_index)[] for _ in 1:nrows]
    for (i, j) in zip(rows_index, cols_index)
        push!(cols_by_rows[i], j)
    end
    return cols_by_rows
end

"""
    _rows_by_cols(rows_index,cols_index)

Returns a vector of columns where each column contains a vector of its row indices.
"""
function _rows_by_cols(rows_index, cols_index)
    return _cols_by_rows(cols_index, rows_index)
end

"""
    matrix2graph(sparse_matrix, [partition_by_rows::Bool=true])

A utility function to generate a graph from input sparse matrix, columns are represented
with vertices and 2 vertices are connected with an edge only if the two columns are mutually
orthogonal.

Note that the sparsity pattern is defined by structural nonzeroes, ie includes explicitly
stored zeros.
"""
function matrix2graph(sparse_matrix::AbstractSparseMatrix{<:Number},
        partition_by_rows::Bool = true)
    (rows_index, cols_index, _) = findnz(sparse_matrix)

    ncols = size(sparse_matrix, 2)
    nrows = size(sparse_matrix, 1)

    num_vtx = partition_by_rows ? nrows : ncols

    inner = SimpleGraph{promote_type(eltype(rows_index), eltype(cols_index))}(num_vtx)

    if partition_by_rows
        rows_by_cols = _rows_by_cols(rows_index, cols_index)
        @inbounds for (cur_row, cur_col) in zip(rows_index, cols_index)
            if !isempty(rows_by_cols[cur_col])
                for next_row in rows_by_cols[cur_col]
                    if next_row < cur_row
                        add_edge!(inner, cur_row, next_row)
                    end
                end
            end
        end
    else
        cols_by_rows = _cols_by_rows(rows_index, cols_index)
        @inbounds for (cur_row, cur_col) in zip(rows_index, cols_index)
            if !isempty(cols_by_rows[cur_row])
                for next_col in cols_by_rows[cur_row]
                    if next_col < cur_col
                        add_edge!(inner, cur_col, next_col)
                    end
                end
            end
        end
    end
    return VSafeGraph(inner)
end

"""
    matrix_colors(A, alg::ColoringAlgorithm = GreedyD1Color();
        partition_by_rows::Bool = false)

Return the colorvec vector for the matrix A using the chosen coloring
algorithm. If a known analytical solution exists, that is used instead.
The coloring defaults to a greedy distance-1 coloring.

Note that if A isa SparseMatrixCSC, the sparsity pattern is defined by structural nonzeroes,
ie includes explicitly stored zeros.

If `ArrayInterface.fast_matrix_colors(A)` is true, then uses
`ArrayInterface.matrix_colors(A)` to compute the matrix colors.
"""
function ArrayInterface.matrix_colors(A::AbstractMatrix,
        alg::SparseDiffToolsColoringAlgorithm = GreedyD1Color();
        partition_by_rows::Bool = false)

    # If fast algorithm for matrix coloring exists use that
    if !partition_by_rows
        ArrayInterface.fast_matrix_colors(A) && return ArrayInterface.matrix_colors(A)
    else
        A_ = A'
        ArrayInterface.fast_matrix_colors(A_) && return ArrayInterface.matrix_colors(A_)
    end

    _A = A isa SparseMatrixCSC ? A : sparse(A) # Avoid the copy
    A_graph = matrix2graph(_A, partition_by_rows)
    return color_graph(A_graph, alg)
end