# Modelling with Diffsol

Ordinary Differential Equations (ODEs) are a powerful tool for modelling a wide range of physical systems. Unlike purely data-driven models, ODEs are based on the underlying physics, biology, or chemistry of the system being modelled. This makes them particularly useful for predicting the behaviour of a system under conditions that have not been observed. In this section, we will introduce the basics of ODE modelling, and illustrate their use with a series of examples written using the Diffsol crate.

The topics covered in this section are:

- [First Order ODEs](first_order_odes.md): First order ODEs are the simplest type of ODE. Any ODE system can be written as a set of first order ODEs, so libraries like Diffsol are designed such that the user provides their equations in this form.
  - [Example: Population Dynamics](population_dynamics.md): A simple example of a first order ODE system, modelling the interaction of predator and prey populations.
- [Higher Order ODEs](higher_order_odes.md): Higher order ODEs are equations that involve derivatives of order greater than one. These can be converted to a system of first order ODEs, which is the form that Diffsol expects.
  - [Example: Spring-mass systems](spring_mass_systems.md): A simple example of a higher order ODE system, modelling the motion of a damped spring-mass system.
- [Discrete Events](discrete_events.md): Discrete events are events that occur at specific times or when the system is in a particular state, rather than continuously. These can be modelled by treating the events as changes in the ODE system's state. Diffsol provides an API to detect and handle these events.
  - [Example: Compartmental models of Drug Delivery](compartmental_models_of_drug_delivery.md): Pharmacokinetic models describe how a drug is absorbed, distributed, metabolised, and excreted by the body. They are a common example of systems with discrete events, as the drug is often administered at discrete times.
  - [Example: Bouncing Ball](bouncing_ball.md): A simple example of a system where the discrete event occurs when the ball hits the ground, instead of at a specific time.
- [DAEs via the Mass Matrix](the_mass_matrix.md): Differential Algebraic Equations (DAEs) are a generalisation of ODEs that include algebraic equations as well as differential equations. Diffsol can solve DAEs by treating them as ODEs with a mass matrix. This section explains how to use the mass matrix to solve DAEs.
  - [Example: Electrical Circuits](electrical_circuits.md): Electrical circuits are a common example of DAEs, here we will model a simple low-pass LRC filter circuit.
- [PDEs](pdes.md): Partial Differential Equations (PDEs) are a generalisation of ODEs that involve derivatives with respect to more than one variable (e.g. a spatial variable). Diffsol can be used to solver PDEs using the method of lines, where the spatial derivatives are discretised to form a system of ODEs.
  - [Example: Heat Equation](heat_equation.md): The heat equation describes how heat diffuses in a domain over time. We will solve the heat equation in a 1D domain with Dirichlet boundary conditions.
  - [Example: Physics-based Battery Simulation](physics_based_battery_simulation.md): A more complex example of a PDE system, modelling the charge and discharge of a lithium-ion battery. For this example we will use the PyBaMM library to form the ODE system, and Diffsol to solve it.

- [Forward Sensitivity Analysis](forward_sensitivity_analysis.md): Sensitivity analysis is a technique used to determine how the output of a model changes with respect to changes in the model parameters. Forward sensitivity analysis calculates the sensitivity of the model output with respect to the parameters by solving the ODE system and the sensitivity equations simultaneously.
  - [Example: Fitting a predator-prey model to data](population_dynamics_fitting.md): An example of fitting a predator-prey model to synthetic data using forward sensitivity analysis.
- [Backwards Sensitivity Analysis](backwards_sensitivity_analysis.md): Backwards sensitivity analysis calculates the sensitivity of a loss function with respect to the parameters by first solving the ODE system and then the adjoint equations backwards in time. This is useful if your model has a high number of parameters, as it can be more efficient than forward sensitivity analysis.
  - [Example: Fitting a spring-mass model to data](spring_mass_fitting.md): An example of fitting a spring-mass model to synthetic data using backwards sensitivity analysis.
  - [Example: Weather prediction using neural ODEs](weather_neural_ode.md): An example of fitting a neural ODE to weather data using backwards sensitivity analysis.
