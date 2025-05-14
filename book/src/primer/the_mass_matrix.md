# DAEs via the Mass Matrix

Differential-algebraic equations (DAEs) are a generalisation of ordinary differential equations (ODEs) that include algebraic equations, or equations that do not involve derivatives. Algebraic equations can arise in many physical systems and often are used to model constraints on the system, such as conservation laws or other relationships between the state variables. For example, in an electrical circuit, the current flowing into a node must equal the current flowing out of the node, which can be written as an algebraic equation.

DAEs can be written in the general implicit form:

\\[
\mathbf{F}(\mathbf{y}, \mathbf{y}', t) = 0
\\]

where \\(\mathbf{y}\\) is the vector of state variables, \\(\mathbf{y}'\\) is the vector of derivatives of the state variables, and \\(\mathbf{F}\\) is a vector-valued function that describes the system of equations. However, for the purposes of this primer and the capabilities of Diffsol, we will focus on a specific form of DAEs called index-1 or semi-explicit DAEs, which can be written as a combination of differential and algebraic equations:

\\[
\begin{align*}
\frac{d\mathbf{y}}{dt} &= \mathbf{f}(\mathbf{y}, t) \\\\
0 &= \mathbf{g}(\mathbf{y}, t)
\end{align*}
\\]

where \\(\mathbf{f}\\) is the vector-valued function that describes the differential equations and \\(\mathbf{g}\\) is the vector-valued function that describes the algebraic equations. The key difference between DAEs and ODEs is that DAEs include algebraic equations that must be satisfied at each time step, in addition to the differential equations that describe the rate of change of the state variables.

How does this relate to the standard form of an explicit ODE that we have seen before? Recall that an explicit ODE can be written as:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

Lets update this equation to include a matrix \\(\mathbf{M}\\) that multiplies the derivative term:

\\[
M \frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

When \\(M\\) is the identity matrix (i.e. a matrix with ones along the diagonal), this reduces to the standard form of an explicit ODE. However, when \\(M\\) has diagonal entries that are zero, this introduces algebraic equations into the system and it reduces to the semi-explicit DAE equations show above. The matrix \\(M\\) is called the *mass matrix*. 

Thus, we now have a general form of a set of differential equations, that includes both ODEs and semi-explicit DAEs. This general form is used by Diffsol to allow users to specify a wide range of problems, from simple ODEs to more complex DAEs. In the next section, we will look at a few examples of DAEs and how to solve them using Diffsol and a mass matrix.

