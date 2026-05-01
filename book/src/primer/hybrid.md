# Hybrid ODEs

Thus far we have considered ODEs that have a single behaviour over time and have introduced discrete events as a way to model systems that have sudden changes in behaviour. However, many systems have multiple modes of behaviour that are active at different times, which we will refer to as hybrid ODEs. These modes could be different discrete events (e.g. dosing protocols) or could be entirely different ODE dynamics (e.g. an epidemic model with policy switching).

In order to model these systems, we will extend our first order ODE formulation to encompass a *set* of ODEs that are indexed by a discrete variable \(N\). This variable can be thought of as a "model index" that determines which ODE is active at any given time. The dynamics of the system are then governed by the following equations:

\\[
\begin{align*}
\frac{d\mathbf{y}}{dt} &= \mathbf{f}_N(\mathbf{y}, t, \mathbf{p}) \\\\
\mathbf{y}(0) &= \mathbf{y}_0
\end{align*}
\\]

with stopping and reset conditions defined by the functions \\(\mathbf{r}_N\\) and \\(\mathbf{s}_N\\) as follows:

\\[
\mathbf{y}^+ = \mathbf{r}_N(\mathbf{y}^-, t_e, \mathbf{p}), \quad \text{when} \quad 0 = \mathbf{s}_N(\mathbf{y}, t_e, \mathbf{p})
\\]

With this in place, we only have to decide on the rules for switching between the different ODEs using the model index \\(N\\). There are many ways to do this, and diffsol is flexible enough to allow users to implement their own switching rules. However, one approach (which is the approach used in diffsol wrappers like diffsol-c and pydiffsol), is to start (i.e. at time \\(t=0\\)) with \\(N=0\\) and then to switch to the next ODE (i.e. \\(N=i\\)) based on the index of the stopping condition that is satisfied (note that \\(\mathbf{s}_N\\) is a vector of stopping conditions). This process is repeated until we have reached the end of the time interval of interest.

To illustrate this using some concrete diffsl examples, we will consider a few examples of hybrid ODEs in the next sections, including a simple dosing protocol and an epidemic model with policy switching.
