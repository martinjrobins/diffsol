# Discrete Events

ODEs describe the continuous evolution of a system over time, but many systems also involve discrete events that occur at specific times. For example, in a compartmental model of drug delivery, the administration of a drug is a discrete event that occurs at a specific time. In a bouncing ball model, the collision of the ball with the ground is a discrete event that changes the state of the system. It is normally difficult to model these events using ODEs alone, as they require a different approach to handle the discontinuities in the system. While we can represent discrete events mathematically using delta functions, many ODE solvers are not designed to handle discontinuities, and may produce inaccurate results or fail to converge during the integration.

Diffsol provides two ways to model discrete events in a system of ODEs:

1. Use the event handling feature to detect when a discrete event occurs, and then manually update the state of the system in Rust. This is a straightforward procedural approach, but if you want to calculate sensitivities with respect to the parameters, you need to propagate the event correction yourself.
2. Use the DiffSL language to specify the `stop` conditions and `reset` actions that occur when a discrete event happens. This is a declarative approach that allows you to describe mathematically *when* a discrete event occurs and *what* happens when it does.

## Procedural Approach

Diffsol allows you to manipulate the internal state of each solver during the time-stepping. Each solver has an internal state that holds information such as the current time \\(t\\), the current state of the system \\(\mathbf{y}\\), and other solver-specific information. When a discrete event occurs, the user can update the internal state of the solver to reflect the change in the system, and then continue the integration of the ODE as normal.

Diffsol also provides a way to stop the integration of the ODEs, either at a specific time or when a specific condition is met, by defining a function \\(g\\) that is equal to zero when the event occurs. This can be useful for modelling systems with discrete events, as it allows the user to control the integration of the ODEs and to handle the events in a flexible way.

\\[
g(\mathbf{y}, t, \mathbf{p}) = 0
\\]

In DiffSL, the `stop` tensor is used to define the event function \\(g\\). The [Solving the Problem](../solve/solving_the_problem.md) and [Root Finding](../specify/closure/root_finding.md) sections provide an introduction to the API for solving ODEs and detecting events with Diffsol.

## Declarative Approach

Rather than writing the event logic directly in Rust, it is also possible to write down the mathematical conditions for when an event occurs and what happens when it does. To define when an event occurs, we re-introduce a stop condition that triggers when the following condition is met:

\\[
g(\mathbf{y}, t_e, \mathbf{p}) = 0
\\]

We can then specify that, when this condition is met, the state of the system should be updated according to a reset function:

\\[
\mathbf{y}^+ = \mathbf{r}(\mathbf{y}^-, t_e, \mathbf{p})
\\]

In DiffSL, the `stop` tensor stores one or more event functions, while the `reset` tensor stores the post-event state. The `reset` tensor must have the same shape as the state tensor `u`, while the `stop` tensor can contain any number of scalar event conditions. 

When a stop condition fires, the solver will stop the integration and return control to the caller. The caller can then apply the reset function to update the state of the system, and then resume the integration of the ODEs. This allows us to model discrete events in a way that is mathematically consistent with the continuous evolution of the system, and also allows us to propagate sensitivities/adjoints through the event in a way that is consistent with the mathematical description of the system.

In the next two sections, we revisit compartmental models of drug delivery and bouncing-ball dynamics and show both the procedural and declarative versions side by side.
