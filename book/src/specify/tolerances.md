# Tolerances

All the solvers in diffsol aim to keep the Local Truncation Error (LTE) (the error accumulated during a single step) below a certain threshold,
using the tolerances that have been set for that particular problem.

The particular error norm that diffsol uses is a weighted root mean squared error

$$
|\delta| = \sum_i \left ( \frac{\delta_i}{|y_i| \cdot rtol + atol_i} \right) ^2
$$

So that the relative tolerance `rtol` defines the error tolerance relative to the current size of the state, and the absolute tolerance `atol` takes over when the state approaches zero.
Note that `atol` is a vector since the scales of different state values might vary, but in practice is is common to set all the elements of `atol` to the same value,
so the builder method [`OdeBuilder::atol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.atol) will accept a single value.

## Output error control

When integrating an output function over time, as well as the ODE state, diffsol allows you to set the step size based on the both the error in the states and output integrals.
The same norm is used for the output error as the states, and the final error is calculated as the maximum of the two individual contributions.

To enable output error control, you need to set the output error tolerances using [`OdeBuilder::out_rtol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.out_rtol)
and [`OdeBuilder::out_atol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.out_atol).
If these tolerances have previously been set and you wish to turn *off* the output error control you can use [`OdeBuilder::turn_off_output_error_control`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.turn_off_output_error_control)

## Forward sensitivity error control

There are `n_p` sets of forward sensitivity equations to be integrated, where `n_p` is the number of parameters, with each set being composed of `n_y` equations.
If forward sensitivity error control is enabled, a separate error norm is calculated for each set,
and these are combined so that the final error is the maximum over the error norms for the base equations, the `n_p` sets of sensitivity equations, and
any other error norms active.

To enable forward sensitivity error control, you need to set the appropriate error tolerances [`OdeBuilder::sens_rtol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.sens_rtol)
and [`OdeBuilder::sens_atol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.sens_atol). You can turn it off
using [`OdeBuilder::turn_off_sensitivities_error_control`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.turn_off_sensitivities_error_control).

It is possible that the scale of different parameters could vary strongly across the parameter vector, in which case it would be benificial
to have a *separate* `sens_atol` for each of the `n_p` sets of sensitivity equations. In this case you can set appropriate scales
for each of the parameters using the [`OdeBuilder::param_scales`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.param_scales) builder method, and
each `sens_atol` for that parameter will be divided by the absolute value of the scale that is set.

## Adjoint sensitivity error control

Calculating the adjoint sensitivities requires both a forward pass, where only the base ODE equations are being solver, and a backwards
pass where the adjoint equations are being solved and, if its a continuous loss function, a set of parameter gradient functions of length `n_p` are also being integrated.
During the forward pass the base error norm and output error control are active, as described above. During the backwards pass
the standard `atol` and `rtol` are used for the adjoint equations, but a separate set of tolerances, [`OdeBuilder::param_rtol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.param_rtol)
and [`OdeBuilder::param_atol`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.param_rtol) are used for the parameter gradients.
As with the other types of error control, if `param_rtol` and `param_atol` are not set, then the parameter gradient equations do not
contribute to the error control.

## Global errors

The LTE is a measure of the error accumulated during a single step of the method. Most importantly, it is *not* a global error, so you might find that
over the duration of a solve the error between your numerical solution and some analytical solution has grown larger than the tolerances that you
set for the problem. Reducing the tolerances, however, will generally also reduce the global error, so you can continue reducing the tolerances
until you achieve the global error that you are aiming for.
