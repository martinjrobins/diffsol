// Translated from https://github.com/JuliaDiff/SparseDiffTools.jl under an MIT license

use petgraph::graph::NodeIndex;

use super::graph::Graph;

/// Returns a vector of rows where each row contains
/// a vector of its column indices.
fn cols_by_rows(non_zeros: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let nrows = if non_zeros.is_empty() {
        0
    } else {
        *non_zeros.iter().map(|(i, _j)| i).max().unwrap() + 1
    };
    let mut cols_by_rows = vec![Vec::new(); nrows];
    for (i, j) in non_zeros.iter() {
        cols_by_rows[*i].push(*j);
    }
    cols_by_rows
}

/// A utility function to generate a graph from input sparse matrix, columns are represented
/// with vertices and 2 vertices are connected with an edge only if the two columns are mutually
/// orthogonal.
///
/// non_zeros: A vector of indices (i, j) where i is the row index and j is the column index
pub fn nonzeros2graph(non_zeros: &[(usize, usize)], ncols: usize) -> Graph {
    let cols_by_rows = cols_by_rows(non_zeros);
    let mut edges = Vec::new();
    for (cur_row, cur_col) in non_zeros.iter() {
        if !cols_by_rows[*cur_row].is_empty() {
            for next_col in cols_by_rows[*cur_row].iter() {
                if next_col < cur_col {
                    edges.push((*cur_col, *next_col));
                }
            }
        }
    }
    let mut graph = Graph::with_capacity(ncols, edges.len());
    for _ in 0..ncols {
        graph.add_node(());
    }
    for (i, j) in edges.iter() {
        graph.add_edge(NodeIndex::new(*i), NodeIndex::new(*j), ());
    }
    graph
}
