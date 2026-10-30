# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.19.0](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.18.0...diffsol-v0.19.0) - 2026-10-30

### Added

- *(diffsol)* add interpolate_vjp ([#425](https://github.com/martinjrobins/diffsol/pull/425))
- *(diffsol)* [**breaking**] improve step_vjp accuracy, add benchmarks ([#422](https://github.com/martinjrobins/diffsol/pull/422))

### Fixed

- default min_timestep to the scalar's epsilon ([#402](https://github.com/martinjrobins/diffsol/pull/402))
- count nonlinear solver failures per step and lower the default limit to 10 ([#397](https://github.com/martinjrobins/diffsol/pull/397))

### Other

- *(diffsol-la)* native impls of ger_cols, speed-ups on faer and nalgebra fill ([#429](https://github.com/martinjrobins/diffsol/pull/429))
- *(diffsol)* consolidate batched/non-batched test_models ([#426](https://github.com/martinjrobins/diffsol/pull/426))

## [0.18.0](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.17.1...diffsol-v0.18.0) - 2026-10-20

### Added

- *(diffsol)* [**breaking**] vjp for runge-kutta ([#413](https://github.com/martinjrobins/diffsol/pull/413))
- *(diffsol)* vjp and jvp of step for OdeSolverMethods ([#409](https://github.com/martinjrobins/diffsol/pull/409))
- *(diffsol-la)* add transposed linear solve ([#406](https://github.com/martinjrobins/diffsol/pull/406))
- *(diffsol-la)* support f32 for cuda-oxide, deprecate C cuda feature ([#387](https://github.com/martinjrobins/diffsol/pull/387))
- *(diffsol-la)* make cuda-oxide containers generic on ScalarCuda ([#386](https://github.com/martinjrobins/diffsol/pull/386))
- *(diffsol-la)* generic cuda-oxide kernels ([#381](https://github.com/martinjrobins/diffsol/pull/381))

### Fixed

- *(ic)* re-evaluate the ic jacobian at the current iterate ([#392](https://github.com/martinjrobins/diffsol/pull/392))

### Other

- *(diffsol-la)* [**breaking**] remove old nalgebra LU solver keeping only non-allocating LU ([#417](https://github.com/martinjrobins/diffsol/pull/417))
- *(readme)* add closure example ([#388](https://github.com/martinjrobins/diffsol/pull/388))
- *(contributing)* add LLM summary to readme

## [0.17.1](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.17.0...diffsol-v0.17.1) - 2026-09-28

### Added

- *(docs.rs)* remove llvm feature on docs build to fix OOM ([#379](https://github.com/martinjrobins/diffsol/pull/379))

## [0.17.0](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.16.2...diffsol-v0.17.0) - 2026-09-21

### Added

- *(diffsol-la)* [**breaking**] for_each_batch, for_each_elem + device/host variants ([#375](https://github.com/martinjrobins/diffsol/pull/375))
- *(diffsol-la)* new cuda-oxide LA backend ([#374](https://github.com/martinjrobins/diffsol/pull/374))
- *(diffsol)* [**breaking**] sensitivity/adjoints calculated by batching ([#373](https://github.com/martinjrobins/diffsol/pull/373))
- *(diffsol-la)* [**breaking**] allow grouped batching B -> C * B, owned side has priority, then lhs ([#370](https://github.com/martinjrobins/diffsol/pull/370))
- *(diffsol)* [**breaking**] refactor Tableau and Bdf to use new SmallVec and SmallMat ([#369](https://github.com/martinjrobins/diffsol/pull/369))
- *(diffsol-la)* add batching for cpu backends ([#361](https://github.com/martinjrobins/diffsol/pull/361))

### Fixed

- use black_blox to prevent benchmark over-optimisation ([#359](https://github.com/martinjrobins/diffsol/pull/359))

### Other

- examples for cpu + gpu parallel ensembles, example for batching and book sections ([#377](https://github.com/martinjrobins/diffsol/pull/377))
- remove unused colog
- *(diffsol-la)* use native faer matmul for sparse gemv ([#367](https://github.com/martinjrobins/diffsol/pull/367))
- *(diffsol)* add a fast pow for common exponents in pi controller ([#365](https://github.com/martinjrobins/diffsol/pull/365))
- *(diffsol-la)* non-allocating nalgebra lu solver ([#364](https://github.com/martinjrobins/diffsol/pull/364))
- *(diffsol-la)* use new rustc algebraic floating point ops for reductions (feature gated for now) ([#363](https://github.com/martinjrobins/diffsol/pull/363))
- *(diffsol-la)* replace column_axpy kernel with fused backwards_diff_update ([#362](https://github.com/martinjrobins/diffsol/pull/362))

## [0.16.2](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.16.1...diffsol-v0.16.2) - 2026-08-18

### Added

- autodiff closures and examples ([#302](https://github.com/martinjrobins/diffsol/pull/302))
- add discrete event performance example and book section ([#338](https://github.com/martinjrobins/diffsol/pull/338))

### Fixed

- use parameter-scaled sens_atol for forward sensitivity ([#339](https://github.com/martinjrobins/diffsol/pull/339))
- consistent capitalisation of diffsol ([#336](https://github.com/martinjrobins/diffsol/pull/336))

### Other

- bdf uses LS param instead of NLS, always uses NewtonNonlinearSolver ([#332](https://github.com/martinjrobins/diffsol/pull/332))
