# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.17.0](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.16.2...diffsol-v0.17.0) - 2026-09-05

### Added

- *(diffsol)* [**breaking**] sensitivity/adjoints calculated by batching ([#373](https://github.com/martinjrobins/diffsol/pull/373))
- *(diffsol-la)* [**breaking**] allow grouped batching B -> C * B, owned side has priority, then lhs ([#370](https://github.com/martinjrobins/diffsol/pull/370))
- *(diffsol)* [**breaking**] refactor Tableau and Bdf to use new SmallVec and SmallMat ([#369](https://github.com/martinjrobins/diffsol/pull/369))
- *(diffsol-la)* add batching for cpu backends ([#361](https://github.com/martinjrobins/diffsol/pull/361))

### Fixed

- use black_blox to prevent benchmark over-optimisation ([#359](https://github.com/martinjrobins/diffsol/pull/359))

### Other

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
