# Example: Bouncing Ball

Modelling a bouncing ball is a simple and intuitive example of a system with discrete events. The ball is dropped from a height \\(h\\) and bounces off the ground with a coefficient of restitution \\(e\\). When the ball hits the ground, its velocity is reversed and scaled by the coefficient of restitution, and the ball rises and then continues to fall until it hits the ground again. This process repeats until halted.

The second order ODE that describes the motion of the ball is given by:

\\[
\frac{d^2x}{dt^2} = -g
\\]

where \\(x\\) is the position of the ball, \\(t\\) is time, and \\(g\\) is the acceleration due to gravity. We can rewrite this as a system of two first order ODEs by introducing a new variable for the velocity of the ball:

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -g
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the ball. This is a system of two first order ODEs, which can be written in vector form as:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where

\\[
\mathbf{y} = \begin{bmatrix} x \\\\ v \end{bmatrix}
\\]

and

\\[
\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} v \\\\ -g \end{bmatrix}
\\]

The initial conditions for the ball, including the height from which it is dropped and its initial velocity, are given by:

\\[
\mathbf{y}(0) = \begin{bmatrix} h \\\\ 0 \end{bmatrix}
\\]

When the ball hits the ground, we need to update the velocity of the ball according to the coefficient of restitution, which is the ratio of the velocity after the bounce to the velocity before the bounce. The velocity after the bounce \\(v'\\) is given by:

\\[
v' = -e v
\\]

where \\(e\\) is the coefficient of restitution. However, to implement this in our ODE solver, we need to detect when the ball hits the ground. We can do this by using Diffsol's event handling feature, which allows us to specify a function that is equal to zero when the event occurs, i.e. when the ball hits the ground. This function \\(g(\mathbf{y}, t)\\) is called an event or root function, and for our bouncing ball problem, it is given by:

\\[
g(\mathbf{y}, t) = x
\\]

where \\(x\\) is the position of the ball. When the ball hits the ground, the event function will be zero and Diffsol will stop the integration, and we can update the velocity of the ball accordingly.

In code, the bouncing ball problem can be solved using Diffsol as follows:

```rust,ignore
{{#include ../../../examples/bouncing-ball/src/main.rs}}
```

{{#include images/bouncing-ball.html}}
