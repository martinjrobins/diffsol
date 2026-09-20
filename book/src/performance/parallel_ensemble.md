# Solving Many ODEs in Parallel

In areas such as model fitting, uncertainty quantification, and population models it is often necessary to solve many ODEs in parallel. Here the goal is to use the parallelism available in your computer to solve many ODEs simultaneously, rather than solving them one after the other. This can be done on both CPUs and GPUs, and the following sections will explore how to achieve this with diffsol

- [On the CPU](./parallel_ensemble_cpu.md)
- [On the GPU](./parallel_ensemble_gpu.md)
