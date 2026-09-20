//! How uncertain electricity demand affects grid frequency.
//!
//! Each sample is an independent power grid with its own demand at one bus. On the GPU the whole
//! ensemble is a single batched solve; on the CPU it is one solve per rayon task.
//!
//! Everything but the model lives in [`ensemble`], which is compiled only with the `cuda-oxide`
//! feature -- so the gating is here, once, rather than on every item.

// the model is always compiled, so it stays under clippy and its test keeps running in a plain
// `cargo test`; only `ensemble` consumes it, and that is feature-gated
#[cfg_attr(not(feature = "cuda-oxide"), allow(dead_code))]
mod swing;

#[cfg(feature = "cuda-oxide")]
mod ensemble;

#[cfg(feature = "cuda-oxide")]
fn main() {
    ensemble::run();
}

#[cfg(not(feature = "cuda-oxide"))]
fn main() {
    println!("this example needs a GPU: build and run it with `just gpu-ensemble`");
}
