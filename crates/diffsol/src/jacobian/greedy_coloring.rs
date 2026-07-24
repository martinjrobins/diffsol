// Translated from https://github.com/JuliaDiff/SparseDiffTools.jl under an MIT license

use petgraph::graph::NodeIndex;

use super::graph::Graph;

/// Greedy graph coloring algorithm.
///
/// Find a coloring of a given input graph such that
/// no two vertices connected by an edge have the same
/// color using greedy approach. The number of colors
/// used may be equal or greater than the chromatic
/// number `χ(G)` of the graph.
pub fn color_graph_greedy(graph: &Graph) -> Vec<usize> {
    let mut result = vec![0; graph.node_count()];
    result[0] = 1;
    let mut available = vec![false; graph.node_count()];

    for ii in 1..graph.node_count() {
        for j in graph.neighbors(NodeIndex::new(ii)) {
            if result[j.index()] != 0 {
                available[result[j.index()] - 1] = true;
            }
        }
        for (i, a) in available.iter().enumerate() {
            if !a {
                result[ii] = i + 1;
                break;
            }
        }
        available.iter_mut().for_each(|x| *x = false);
    }
    result
}
