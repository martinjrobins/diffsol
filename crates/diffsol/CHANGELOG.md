# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.16.3](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.16.2...diffsol-v0.16.3) - 2026-08-20

### Fixed

- use black_blox to prevent benchmark over-optimisation ([#359](https://github.com/martinjrobins/diffsol/pull/359))

## [0.16.2](https://github.com/martinjrobins/diffsol/compare/diffsol-v0.16.1...diffsol-v0.16.2) - 2026-08-18

### Added

- autodiff closures and examples ([#302](https://github.com/martinjrobins/diffsol/pull/302))
- add discrete event performance example and book section ([#338](https://github.com/martinjrobins/diffsol/pull/338))

### Fixed

- use parameter-scaled sens_atol for forward sensitivity ([#339](https://github.com/martinjrobins/diffsol/pull/339))
- consistent capitalisation of diffsol ([#336](https://github.com/martinjrobins/diffsol/pull/336))

### Other

- bdf uses LS param instead of NLS, always uses NewtonNonlinearSolver ([#332](https://github.com/martinjrobins/diffsol/pull/332))
