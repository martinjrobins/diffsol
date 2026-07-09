# Stage 1: Build
FROM rust:1.89-slim-bookworm AS builder

WORKDIR /usr/src/diffsol

# Copy workspace Cargo.toml and scope to only diffsol and its diffsol-la and
# diffsol-nl dependencies (avoids pulling in diffsol-c and examples/* which
# aren't needed for benchmarks)
COPY Cargo.toml .
RUN sed -i 's/^members = .*/members = ["diffsol-la", "diffsol-nl", "diffsol"]/' Cargo.toml && \
    sed -i 's/^default-members = .*/default-members = ["diffsol-la", "diffsol-nl", "diffsol"]/' Cargo.toml

# Copy crate Cargo.toml files for dependency resolution
COPY diffsol/Cargo.toml diffsol/
COPY diffsol-la/Cargo.toml diffsol-la/
COPY diffsol-nl/Cargo.toml diffsol-nl/

# Create minimal stubs so cargo can resolve and pre-compile deps
RUN mkdir -p diffsol/src diffsol/benches diffsol-la/src diffsol-nl/src \
    && echo '' > diffsol/src/lib.rs \
    && echo '' > diffsol-la/src/lib.rs \
    && echo '' > diffsol-nl/src/lib.rs \
    && echo 'fn main() {}' > diffsol/benches/ode_solvers_ci.rs \
    && echo 'fn main() {}' > diffsol/benches/ode_solvers.rs \
    && echo 'fn main() {}' > diffsol/benches/lin_alg_ops.rs \
    && echo 'fn main() {}' > diffsol/benches/pybamm_dfn.rs

# Pre-compile dependencies (cached layer — only rebuilt when Cargo.toml changes)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --lib --release

# Copy real source and build the benchmark binary
COPY diffsol-la/ diffsol-la/
COPY diffsol-nl/ diffsol-nl/
COPY diffsol/ diffsol/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --bench ode_solvers_ci --release && \
    for p in target/release/ode_solvers_ci target/release/deps/ode_solvers_ci-*; do \
      [ -f "$p" ] && [ -x "$p" ] && cp "$p" /usr/local/bin/ode_solvers_ci && break; \
    done && \
    chmod +x /usr/local/bin/ode_solvers_ci

# Stage 2: Runtime (~74 MB base)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/ode_solvers_ci /usr/local/bin/

CMD ["ode_solvers_ci", "--bench"]
