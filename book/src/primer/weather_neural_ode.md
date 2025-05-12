# Example: Weather prediction using Neural ODEs

If we wish to train a neural network to predict the weather or any other time series dataset, we can use a Neural ODE. A Neural ODE replaces the rhs of an ODE with a neural network.

$$
\frac{dy}{dt} = f(y, p)
$$

Here, \\(y(t)\\) is the state of the system at time \\(t\\), and \\(f\\) is a neural network with parameters \\(p\\). The neural network is trained to predict the derivative of the state, and the ODE solver is used to integrate the state forward in time, and to calculate gradients of the loss function with respect to the parameters of the neural network.

In this example, we will duplicate the weather prediction example from the excellent [blog post](https://sebastiancallh.github.io/post/neural-ode-weather-forecast/) by Sebastian Callh, but instead using Diffsol as the solver. We'll skip over some of the details, but you can read more details about the problem setup in the original blog post, and see the full code in the [Diffsol repository](https://github.com/martinjrobins/diffsol).

First we'll need a neural network model, and we'll use [Equinox](https://github.com/patrick-kidger/equinox) and [JAX](https://docs.jax.dev/en/latest/index.html) for this. We'll define a simple neural network with 3 layers like so

```python
{{#include ../../../examples/neural-ode-weather-prediction/src/model/model.py:39:54}}
```

We will then create four JAX functions that will allow us to calculate:

- the rhs function \\(f(y, p)\\) of the Neural ODE, where \\(y\\) is the state of the system and \\(p\\) are the parameters.
- the Jacobian-vector product of the rhs function with respect to the state \\(y\\).
- the negative vector-Jacobian product of the rhs function with respect to the state \\(y\\).
- the negative vector-Jacobian product of the rhs function with respect to the parameters \\(p\\).

We will need all four of these to define the ODE problem and to solve it using Diffsol.

```python
{{#include ../../../examples/neural-ode-weather-prediction/src/model/model.py:57:80}}
```

Finally, we can export all four of these JAX functions to ONNX, which will allow us to use them within rust.

```python
{{#include ../../../examples/neural-ode-weather-prediction/src/model/model.py:25:33}}
```

Within rust now, we can define a Diffsol system of equations by creating a struct `NeuralOde`. We'll use the [`ort`](https://ort.pyke.io/) crate and the ONNX Runtime to load the ONNX models that we made in Python.

```rust,ignore
{{#include ../../../examples/neural-ode-weather-prediction/src/main.rs:33:85}}
```

We'll also implement the `OdeSystemAdjoint` trait for `NeuralOde`, which will allow us to use the adjoint method to calculate gradients of out loss function with respect to the parameters of the neural network. As an example, here is the implementation of the `NonLinearOp` trait:

```rust,ignore
{{#include ../../../examples/neural-ode-weather-prediction/src/main.rs:196:220}}
```

We'll also need an optimiser, so we'll write an AdamW algorithm using the definition in the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) as a guide:

```rust,ignore
{{#include ../../../examples/neural-ode-weather-prediction/src/main.rs:318:374}}
```

We'll then define our loss function, which will return the sum of squared errors between the solution and the data points, along with the gradients of the loss function with respect to the parameters. Since the size of the parameter vector is quite large (>2000), we'll use the adjoint method to calculate the gradients.

```rust,ignore
{{#include ../../../examples/neural-ode-weather-prediction/src/main.rs:376:397}}
```

Finally, we can train the neural network to predict the weather. Following the example given in the linked blog post above, we'll train in stages by increasing the number of datapoints by four each time. Each time we'll train for 150 steps using the AdamW optimiser.

```rust,ignore
{{#include ../../../examples/neural-ode-weather-prediction/src/main.rs:408:427}}
```

To give an indication of the results, we'll plot the results after we've used the first 20 data-points to train the model, and we'll predict the model solution to the entire dataset.

{{#include ../../../examples/neural-ode-weather-prediction/neural-ode-weather_5}}

This seems to work well, and is good at matching the data points a long way into the future. This has been a whirlwind description of both Neural ODEs and this particular analysis. For a more detailed explanation, please refer to the original blog post by Sebastian Callh. We've also skipped over many more boring parts of the code, and you can see the full code for this example in the [Diffsol repository](https://github.com/martinjrobins/diffsol).
