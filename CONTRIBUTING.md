# Contributing to diffsol

Thank you for your interest in contributing to diffsol! This guide will help you get started with contributing, and is written for someone new to contributions to a Rust crate. If you are already familiar with Rust development and contributing to open-source libraries on Github you can skip over most of the sections, however the "Setting Up Your Development Environment" section may still be useful as it includes information on optional dependencies required for certain features of diffsol.

## Table of Contents

- [Getting Started](#getting-started)
  - [Forking the Repository](#forking-the-repository)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Building the Project](#building-the-project)
- [Making Changes](#making-changes)
  - [Creating a Branch](#creating-a-branch)
  - [Writing Code](#writing-code)
  - [Code Style and Formatting](#code-style-and-formatting)
- [Testing](#testing)
  - [Running Tests](#running-tests)
  - [Writing Tests](#writing-tests)
  - [Code Coverage](#code-coverage)
- [Documentation](#documentation)
- [Submitting Your Contribution](#submitting-your-contribution)
  - [Committing Your Changes](#committing-your-changes)
  - [Opening a Pull Request](#opening-a-pull-request)
- [Review Process](#review-process)
- [Getting Help](#getting-help)

## Getting Started

### Diffsol repositories

The diffsol ecosystem consists of several repositories. Before contributing, please ensure you route your issue or pull request to the appropriate repository:

- [diffsol](https://github.com/martinjrobins/diffsol) - the core ODE solver library
- [diffsl](https://github.com/martinjrobins/diffsl) - the DiffSL DSL compiler and JIT backends
- [pydiffsol](https://github.com/alexallmont/pydiffsol) - Python bindings

This guide focuses on contributing to the core diffsol library. Many of the principles apply to the other repositories as well, but please check their individual contributing guidelines for any differences.

### Forking the Repository

1. Navigate to the [diffsol repository](https://github.com/martinjrobins/diffsol) on GitHub.
2. Click the **"Fork"** button in the upper right corner of the page.
3. This creates a copy of the repository in your GitHub account.

### Setting Up Your Development Environment

#### Prerequisites

- **Rust**: Install Rust using [rustup](https://rustup.rs/):

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

  After installation, ensure you have the latest stable version:

  ```bash
  rustup update stable
  ```

- **Git**: Ensure you have Git installed. You can download it from [git-scm.com](https://git-scm.com/).

#### Clone Your Fork

Clone your forked repository to your local machine:

```bash
git clone https://github.com/YOUR-USERNAME/diffsol.git
cd diffsol
```

Replace `YOUR-USERNAME` with your GitHub username.

#### Add Upstream Remote

Add the original repository as an upstream remote to keep your fork synchronized:

```bash
git remote add upstream https://github.com/martinjrobins/diffsol.git
```

#### Install Optional Dependencies

Diffsol has several optional features that may require additional dependencies:

- **SuiteSparse KLU** (for `suitesparse` feature):
  - On Ubuntu/Debian: `sudo apt-get install libsuitesparse-dev`
  - On macOS: `brew install suite-sparse`

- **LLVM** (for `diffsl-llvm15`, `diffsl-llvm16`, `diffsl-llvm17`, `diffsl-llvm18` features):
  - Install LLVM 15, 16, 17, or 18 depending on which feature you want to use
  - On Ubuntu/Debian: Use LLVM install script

    ```bash
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh <version number>
    ```

  - On macOS: `brew install llvm@<version>`
  - [Build from source](https://llvm.org/docs/GettingStarted.html) if necessary.
  - You will need to set the `LLVM_SYS_XXX_PREFIX` (see [`llvm-sys`](https://gitlab.com/taricorp/llvm-sys.rs)) and `LLVM_DIR` environment variables to point to your LLVM installation, where `XXX` is the version number (`150`, `160`, `170` or `181`). e.g.

    ```bash
    export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
    export LLVM_DIR=/usr/lib/llvm-17
    ```

### Building the Project

Build the project with default features:

```bash
cargo build
```

Or build with specific features (e.g., with DiffSL Cranelift backend):

```bash
cargo build --features diffsl-cranelift
```

Or build disabling one or more default features:

```bash
cargo build --no-default-features --features "faer"
```

To build with all features:

```bash
cargo build --all-features
```

## Making Changes

### Creating a Branch

If you have forked the repository you can work in the `main` branch of your fork. If you are contributing directly to the diffsol repository, always create a new branch for your work. Use a descriptive name for your branch:

```bash
git checkout -b feat-your-feature-name
```

or

```bash
git checkout -b bug-description
```

Keep your branch up to date with upstream if you are working on a long-lived branch or fork. For a forked repository, use:

```bash
git fetch upstream
git rebase upstream/main
```

For direct contributions, you can use:

```bash
git fetch origin
git rebase origin/main
```

### Writing Code

- **Follow Rust conventions**: Follow standard Rust naming conventions and idioms, see the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- **Keep changes focused**: Make small, focused commits that address a single concern.
- **Add comments**: Document any complex logic with [comments](https://doc.rust-lang.org/book/ch03-04-comments.html) and public APIs with [doc comments](https://doc.rust-lang.org/book/ch14-02-publishing-to-crates-io.html).
- **Add tests**: Include [tests](https://doc.rust-lang.org/book/ch11-00-testing.html) for new functionality or bug fixes.

### Code Style and Formatting

Diffsol uses the standard Rust formatting style enforced by `rustfmt`.

#### Format Your Code

Before committing, format your code using:

```bash
cargo fmt
```

This will automatically format all Rust code in the project according to the standard Rust style.

#### Lint Your Code

Use [Clippy](https://doc.rust-lang.org/clippy/) to catch common mistakes and improve your code:

```bash
cargo clippy
```

If you are using an IDE like VSCode with [Rust Analyzer](https://rust-analyzer.github.io/), it can also run Clippy automatically on save.

## Testing

### Running Tests

Run all tests with:

```bash
cargo test
```

If you are working in an IDE like VSCode with Rust Analyzer, you can also run tests directly from the IDE interface.

Run tests for a specific feature:

```bash
cargo test --features diffsl-cranelift
```

Run a specific test by name:

```bash
cargo test test_name
```

Run tests with output displayed:

```bash
cargo test -- --nocapture
```

Run tests in the main crate only (excluding examples):

```bash
cargo test -p diffsol
```

### Writing Tests

When adding new functionality or fixing bugs, [include tests](https://doc.rust-lang.org/book/ch11-00-testing.html). Diffsol currently only has unit tests and some documentation tests, but please feel free to add integration tests as needed.

#### Unit Tests

Add unit tests in the same file as the code, in a `tests` module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert_eq!(my_function(2), 4);
    }
}
```

#### Integration Tests

Add integration tests in the `tests/` directory at the root of the crate:

```rust
// tests/my_integration_test.rs
use diffsol::*;

#[test]
fn test_integration() {
    // Your integration test here
}
```

#### Documentation Tests

Add examples in doc comments that are automatically tested:

```rust
/// Adds two numbers together.
///
/// # Examples
///
/// ```
/// use diffsol::add;
/// assert_eq!(add(2, 2), 4);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### Code Coverage

The project uses code coverage with [tarpaulin](https://github.com/xd009642/tarpaulin) to ensure tests adequately cover the codebase. CI runs coverage checks automatically on pull requests.

To run coverage locally (requires `cargo-tarpaulin`) with a feature enabled:

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --features diffsl-cranelift
```

## Documentation

### Building Documentation

Build the documentation locally:

```bash
cargo doc --open
```

This builds the API documentation and opens it in your browser.

Build documentation for all features:

```bash
cargo doc --all-features --open
```

### mdBook Documentation

The project also has mdBook documentation in the `book/` directory. To build and view it:

```bash
# Install mdbook if you haven't already
cargo install mdbook

# Build and serve the book
cd book
mdbook serve --open
```

## Submitting Your Contribution

### Committing Your Changes

Prefer clear, descriptive commit messages. PRs with multiple commits may be squashed when merged, so focus on clarity for the ease of review rather than long-term history.

E.g., for a new feature:

```bash
git add .
git commit -m "feat: brief description of the feature"
```

For bug fixes:

```bash
git commit -m "fix: brief description of the fix"
```

Use conventional commit format when possible:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding or updating tests
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

### Opening a Pull Request

1. **Push your branch** to your fork:

   ```bash
   git push origin your-branch-name
   ```

2. **Navigate to your forked repository** on GitHub.

3. **Click "Pull requests"** and then **"New pull request"**.

4. **Select your branch** from the dropdown menu to compare with the `main` branch of the original diffsol repository.

5. **Fill out the pull request** with:
   - A clear title describing the change
   - A short summary of what has been changed and why, you can refer to the issue if applicable for additional context
   - Reference any related issues (e.g., "Fixes #123" or "Relates to #456")

6. **Ensure CI checks pass**:
   - All tests pass
   - Code is properly formatted (`cargo fmt`)
   - No Clippy warnings
   - Code coverage is maintained or improved

7. **Request review** from maintainers. A PR will need at least one approval before it can be merged.

## Getting Help

If you need help or have questions:

- **Open an [issue](https://github.com/martinjrobins/diffsol/issues)**: For bugs, feature requests, or questions about the project.
- **Open a [discussion](https://github.com/martinjrobins/diffsol/discussions)**: For broader topics, questions, or community support.
- **Check documentation**: Review the [API docs](https://docs.rs/diffsol) and [mdBook](https://martinjrobins.github.io/diffsol/).
- **Search existing issues and discussions**: Your question might already be answered.

## Additional Notes

### Performance Testing

If your changes affect performance, consider running benchmarks:

```bash
cargo bench
```

### Checking All Features

Before submitting, ensure your changes work with different feature combinations:

```bash
# Test with default features
cargo test

# Test with faer only
cargo test --no-default-features --features faer

# Test with nalgebra only
cargo test --no-default-features --features nalgebra

# Test with diffsl features
cargo test --features diffsl-cranelift
```