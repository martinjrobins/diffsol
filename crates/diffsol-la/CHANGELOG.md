# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/martinjrobins/diffsol/compare/diffsol-la-v0.1.1...diffsol-la-v0.2.0) - 2026-09-05

### Added

- *(diffsol)* [**breaking**] sensitivity/adjoints calculated by batching ([#373](https://github.com/martinjrobins/diffsol/pull/373))
- *(diffsol-la)* [**breaking**] allow grouped batching B -> C * B, owned side has priority, then lhs ([#370](https://github.com/martinjrobins/diffsol/pull/370))
- *(diffsol)* [**breaking**] refactor Tableau and Bdf to use new SmallVec and SmallMat ([#369](https://github.com/martinjrobins/diffsol/pull/369))
- *(diffsol-la)* add batching for cpu backends ([#361](https://github.com/martinjrobins/diffsol/pull/361))

### Other

- *(diffsol-la)* use native faer matmul for sparse gemv ([#367](https://github.com/martinjrobins/diffsol/pull/367))
- *(diffsol-na)* add fast pow path for newton convergence ([#366](https://github.com/martinjrobins/diffsol/pull/366))
- *(diffsol-la)* non-allocating nalgebra lu solver ([#364](https://github.com/martinjrobins/diffsol/pull/364))
- *(diffsol-la)* use new rustc algebraic floating point ops for reductions (feature gated for now) ([#363](https://github.com/martinjrobins/diffsol/pull/363))
- *(diffsol-la)* replace column_axpy kernel with fused backwards_diff_update ([#362](https://github.com/martinjrobins/diffsol/pull/362))

## [0.1.1](https://github.com/martinjrobins/diffsol/compare/diffsol-la-v0.1.0...diffsol-la-v0.1.1) - 2026-08-18

### Fixed

- look up row indices through `row_idx` in MatrixSparsity::indices ([#342](https://github.com/martinjrobins/diffsol/pull/342))
- consistent capitalisation of diffsol ([#336](https://github.com/martinjrobins/diffsol/pull/336))

### Other

- builder and tolerances sections ([#346](https://github.com/martinjrobins/diffsol/pull/346))
