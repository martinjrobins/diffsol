# WebAssembly

Most of diffsol can be compiled to WebAssembly, allowing it to be used in web applications in the browser or other environments that support Wasm.

The main limitation is that JIT compilation of the DiffSL DSL is not yet supported in Wasm. This means you must use rust closures or the `OdeEquations` trait to specify your problems, rather than using the DiffSL DSL. 

## An example Yew app

To demonstrate using Diffsol in a WebAssembly application, we will create a simple Yew app that simulates a population dynamics model. The app will allow users to move sliders to adjust the parameters of the model and see the results in real-time using a plotly chart.

A [demo](https://martinjrobins.github.io/diffsol/examples/population-dynamics-wasm-yew/) built from this example is available online.

### Getting started

First, you need to add the `wasm32-unknown-unknown` target to your Rust toolchain:

```bash
rustup target add wasm32-unknown-unknown
```

We'll use trunk to build our Yew app, so you need to install it if you haven't already:

```bash
cargo install trunk
```

Then, create a new Yew app:

```bash
cargo new example-yew-diffsol
```

And add the necessary dependencies to your `Cargo.toml`:

```toml
[dependencies]
yew = { version = "0.21.0", features = ["csr"] }
diffsol = { version = "0.6.2", features = [] }
nalgebra = { workspace = true }
yew-plotly = "0.3.0"
web-sys = "0.3.77"
```

### The Yew app

We'll keep it simple and create a single component that will handle the population dynamics simulation. This will consist of the following:

- An `problem` of type `OdeSolverProblem` that we'll create when the component is mounted, and keep wrapped in a `use_mut_ref` hook so we can modify it when the parameters change.
- The initial parameters for the problem held in `params`, which will be stored in a `use_state` hook.
- An `onchange` callback that will update the parameters of the problem when the user changes the sliders.
- A `Plotly` component that will display the results of the simulation.
- Two sliders for the user to adjust two out of the four parameters of the model.

```rust,ignore
{{#include ../../../examples/population-dynamics-wasm-yew/src/main.rs}}
```

### Building and running the app

You can build and run the app using `trunk`:

```bash
trunk serve
```