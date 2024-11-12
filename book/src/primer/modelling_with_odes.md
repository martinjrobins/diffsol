# Modelling with ODEs

Ordinary Differential Equations (ODEs) are a powerful tool for modelling a wide range of physical systems. Unlike purely data-driven models, ODEs are based on the underlying physics, biology, or chemistry of the system being modelled. This makes them particularly useful for predicting the behaviour of a system under conditions that have not been observed before. In this section, we will introduce the basics of ODE modelling, and illustrate their use with a series of examples written using the DiffSol crate.

The topics covered in this section are:
- [First Order ODEs](./primer/first_order_odes.md): First order ODEs are the simplest type of ODE. Any ODE system can be written as a set of first order ODEs, so libraries like DiffSol are designed such that the user provides their equations in this form.
    - [Example: Population Dynamics](./primer/population_dynamics.md): A simple example of a first order ODE system, modelling the interaction of predator and prey populations.
- [Higher Order ODEs](./primer/higher_order_odes.md): Higher order ODEs are ODEs that involve derivatives of order higher than one. These can be converted to a system of first order ODEs, which is the form that DiffSol expects.
    - [Example: Spring-mass systems](./primer/spring_mass_systems.md): A simple example of a higher order ODE system, modelling the motion of a damped spring-mass system.
- [Discrete Events](./primer/discrete_events.md): Discrete events are events that occur at specific times or when the system is in a particular state, rather than continuously. These can be modelled using ODEs by treating the events as changes in the system's state. DiffSol provides an API to detect and handle these events.
    - [Example: Compartmental models of Drug Delivery](./primer/compartmental_models_of_drug_delivery.md): Pharmacokinetic models are a common example of systems with discrete events, where the drug is absorbed, distributed, metabolised, and excreted by the body. The drug is often administered at discrete times, and the model must account for this.
    - [Example: Bouncing Ball](./primer/bouncing_ball.md): A simple example of a system where the discrete event occurs when the ball hits the ground, instead of a specific time.
- [DAEs via the Mass Matrix](./primer/the_mass_matrix.md): Differential Algebraic Equations (DAEs) are a generalisation of ODEs that include algebraic equations as well as differential equations. DiffSol can solve DAEs by treating them as ODEs with a mass matrix. This section explains how to use the mass matrix to solve DAEs.
    - [Example: Electrical Circuits](./primer/electrical_circuits.md): Electrical circuits are a common example of DAEs, here we will model a simple low-pass LRC filter circuit.
    