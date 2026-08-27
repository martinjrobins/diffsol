#pragma once

// Largest column range the small-coefficient kernels accept, and the size of the by-value
// coefficient arrays they take. Must match `MAX_SMALL_COLS` in `src/matrix/mod.rs`.
//
// Two things drive it: the BDF difference table is at most MAX_ORDER + 1 = 6 columns wide, and
// the widest built-in Runge-Kutta tableau (tsit45) has 7 stages. 8 covers both.
#define MAX_SMALL_COLS 8

//
// Shared inline functions for 2D grid kernel launches.
//
// gridDim.y = nbatch (number of batches of the output/first array)
// blockIdx.y = batch index (always < nbatch by construction — no batch guard needed)
//
// Broadcasting is grouped: an operand holding `src_nbatch` batches repeats each of them over
// `gridDim.y / src_nbatch` contiguous output batches, so `src_nbatch == 1` reads batch 0 for
// every output batch and `src_nbatch == B` against `gridDim.y == B * P` reads batch `b / P`.
// Batch counts that are not exact multiples of each other are rejected host-side
// (`Context::assert_broadcastable_into`).
__device__ inline int broadcast_batch(int b, int src_nbatch) {
    return b * src_nbatch / (int)gridDim.y;
}

// (a) Unary: one array, no broadcasting.
// Used by: vec_fill, vec_mul_assign_scalar
__device__ inline bool batch_unary_setup(
    int* b, int* elem, int nstates
) {
    *elem = blockIdx.x * blockDim.x + threadIdx.x;
    *b = blockIdx.y;
    return (*elem < nstates);
}

// (b) Binary: two arrays, only the second (rhs) may broadcast.
// Used by: vec_add_assign, vec_sub_assign, vec_mul_assign, vec_div_assign,
//          vec_copy, vec_axpy, vec_axpy_offset, mat_scale_add_assign
__device__ inline bool batch_binary_setup(
    int* elem, int nstates,
    int* li, int lhs_stride,
    int* ri, int rhs_stride, int rhs_nbatch
) {
    *elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (*elem >= nstates) return false;
    int b = blockIdx.y;
    *li = b * lhs_stride + *elem;
    *ri = broadcast_batch(b, rhs_nbatch) * rhs_stride + *elem;
    return true;
}

// (c) Binary operator with an output array: every operand may broadcast, so the output
// carries the largest batch count and each side is broadcast against it.
// Used by: vec_add, vec_sub
__device__ inline bool batch_binary_op_setup(
    int* elem, int nstates,
    int* li, int lhs_stride, int lhs_nbatch,
    int* ri, int rhs_stride, int rhs_nbatch,
    int* oi, int out_stride, int out_nbatch
) {
    *elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (*elem >= nstates) return false;
    int b = blockIdx.y;
    *li = broadcast_batch(b, lhs_nbatch) * lhs_stride + *elem;
    *ri = broadcast_batch(b, rhs_nbatch) * rhs_stride + *elem;
    *oi = broadcast_batch(b, out_nbatch) * out_stride + *elem;
    return true;
}

// (d) Ternary: three arrays, second and third may broadcast (the first is the destination
// and always carries the launch's batch count).
// Used by: mat_scale_add_assign
__device__ inline bool batch_ternary_setup(
    int* elem, int nstates,
    int* li, int lhs_stride,
    int* ri, int rhs_stride, int rhs_nbatch,
    int* oi, int out_stride, int out_nbatch
) {
    *elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (*elem >= nstates) return false;
    int b = blockIdx.y;
    *li = b * lhs_stride + *elem;
    *ri = broadcast_batch(b, rhs_nbatch) * rhs_stride + *elem;
    *oi = broadcast_batch(b, out_nbatch) * out_stride + *elem;
    return true;
}

// Index-based kernels: iterate over an index array to map positions.

// Gather/scatter: self position = j, other position = indices[j].
// Used by: vec_gather, vec_scatter
__device__ inline bool batch_gather_scatter_setup(
    int* j, int nindices,
    int* si, int self_stride, int self_nbatch,
    int* oi, int other_stride, int other_nbatch,
    const int* __restrict__ indices
) {
    *j = blockIdx.x * blockDim.x + threadIdx.x;
    if (*j >= nindices) return false;
    int b = blockIdx.y;
    int src = indices[*j];
    *si = broadcast_batch(b, self_nbatch) * self_stride + *j;
    *oi = broadcast_batch(b, other_nbatch) * other_stride + src;
    return true;
}

// Assign-at-indices: self position = indices[j], write scalar.
// Used by: vec_assign_at_indices
__device__ inline bool batch_assign_at_setup(
    int* j, int nindices,
    int* si, int self_stride, int self_nbatch,
    const int* __restrict__ indices
) {
    *j = blockIdx.x * blockDim.x + threadIdx.x;
    if (*j >= nindices) return false;
    int b = blockIdx.y;
    int idx = indices[*j];
    *si = broadcast_batch(b, self_nbatch) * self_stride + idx;
    return true;
}

// Copy-by-index: both self and other positions = indices[j].
// Used by: vec_copy_from_indices
__device__ inline bool batch_copy_indices_setup(
    int* j, int nindices,
    int* si, int self_stride, int self_nbatch,
    int* oi, int other_stride, int other_nbatch,
    const int* __restrict__ indices
) {
    *j = blockIdx.x * blockDim.x + threadIdx.x;
    if (*j >= nindices) return false;
    int b = blockIdx.y;
    int idx = indices[*j];
    *si = broadcast_batch(b, self_nbatch) * self_stride + idx;
    *oi = broadcast_batch(b, other_nbatch) * other_stride + idx;
    return true;
}

// Set-data-with-indices: self position = dst_indices[j], other position = src_indices[j].
// Used by: mat_set_data_with_indices
__device__ inline bool batch_set_data_setup(
    int* j, int n,
    int* si, int self_stride, int self_nbatch,
    int* oi, int other_stride, int other_nbatch,
    const int* __restrict__ dst_indices,
    const int* __restrict__ src_indices
) {
    *j = blockIdx.x * blockDim.x + threadIdx.x;
    if (*j >= n) return false;
    int b = blockIdx.y;
    int di = dst_indices[*j];
    int si_idx = src_indices[*j];
    *si = broadcast_batch(b, self_nbatch) * self_stride + di;
    *oi = broadcast_batch(b, other_nbatch) * other_stride + si_idx;
    return true;
}

// Matrix diagonal ops: iterate over nrows, compute column-major diagonal offset.
// Used by: mat_from_diagonal, mat_get_diagonal
__device__ inline bool batch_diagonal_setup(
    int* i, int nrows,
    int* mi, int mat_stride, int mat_nbatch,
    int* di, int diag_stride, int diag_nbatch
) {
    *i = blockIdx.x * blockDim.x + threadIdx.x;
    if (*i >= nrows) return false;
    int b = blockIdx.y;
    *mi = broadcast_batch(b, mat_nbatch) * mat_stride + *i * nrows + *i;
    *di = broadcast_batch(b, diag_nbatch) * diag_stride + *i;
    return true;
}

// Matrix set-column: iterate over nrows, compute column offset.
// Used by: mat_set_column
__device__ inline bool batch_set_column_setup(
    int* i, int nrows,
    int* mi, int mat_stride, int mat_nbatch,
    int* ci, int col_stride, int col_nbatch,
    int column_index
) {
    *i = blockIdx.x * blockDim.x + threadIdx.x;
    if (*i >= nrows) return false;
    int b = blockIdx.y;
    *mi = broadcast_batch(b, mat_nbatch) * mat_stride + column_index * nrows + *i;
    *ci = broadcast_batch(b, col_nbatch) * col_stride + *i;
    return true;
}
