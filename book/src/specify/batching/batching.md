# Batching

All the matrix and vector traits in diffsol are implemented using batching, similar to the batching that you might be used to in several machine learning frameworks. This allows diffsol to solve `nbatch` ODEs in lockstep, which is particularly useful for solving many ODEs in parallel on the GPU.

This section covers configuring batches with `OdeBuilder` and working with batched vectors.
