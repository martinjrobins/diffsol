// Translated from https://github.com/JuliaDiff/SparseDiffTools.jl under an MIT license

use petgraph::graph::NodeIndex;

use crate::matrix::Matrix;

use super::graph::Graph;

/// Returns a vector of rows where each row contains
/// a vector of its column indices.
fn cols_by_rows(rows_index: Vec<usize>, cols_index: Vec<usize>) -> Vec<Vec<usize>> {
    let nrows = if rows_index.is_empty() { 0 } else { *rows_index.iter().max().unwrap() };
    let mut cols_by_rows = vec![Vec::new(); nrows];
    for (i, j) in rows_index.iter().zip(cols_index.iter()) {
        cols_by_rows[*i].push(*j);
    }
    cols_by_rows
}

/// Returns a vector of columns where each column contains a vector of its row indices.
fn rows_by_cols(rows_index: Vec<usize>, cols_index: Vec<usize>) -> Vec<Vec<usize>> {
    cols_by_rows(cols_index, rows_index)
}

/// A utility function to generate a graph from input sparse matrix, columns are represented
/// with vertices and 2 vertices are connected with an edge only if the two columns are mutually
/// orthogonal.
pub fn matrix2graph<M: Matrix>(sparse_matrix: &M, partition_by_rows: bool) -> Graph {
    let (rows_index, cols_index, _) = sparse_matrix.findnz();
    let ncols = sparse_matrix.ncols();
    let nrows = sparse_matrix.nrows();
    let num_vtx = if partition_by_rows { nrows } else { ncols };
    let mut inner = Graph::with_capacity(num_vtx, num_vtx * 2);
    if partition_by_rows {
        let rows_by_cols = rows_by_cols(rows_index, cols_index);
        for (cur_row, cur_col) in rows_index.iter().zip(cols_index.iter()) {
            let cur_row_index = NodeIndex::new(*cur_row);
            if !rows_by_cols[*cur_col].is_empty() {
                for next_row in rows_by_cols[*cur_col].iter() {
                    if next_row < cur_row {
                        let next_row_index = NodeIndex::new(*next_row);
                        inner.add_edge(cur_row_index, next_row_index, ());
                    }
                }
            }
        }
    } else {
        let cols_by_rows = cols_by_rows(rows_index, cols_index);
        for (cur_row, cur_col) in rows_index.iter().zip(cols_index.iter()) {
            let cur_col_index = NodeIndex::new(*cur_col);
            if !cols_by_rows[*cur_row].is_empty() {
                for next_col in cols_by_rows[*cur_row].iter() {
                    if next_col < cur_col {
                        let next_col_index = NodeIndex::new(*next_col);
                        inner.add_edge(cur_col_index, next_col_index, ());
                    }
                }
            }
        }
    }
    inner
}