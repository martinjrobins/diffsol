# Example: Compartmental models of Drug Delivery

# Pharmokinetic Modelling

The field of Pharmacokinetics (PK) provides a quantitative basis for describing the delivery of a drug to a patient, the diffusion of that drug through the plasma/body tissue, and the subsequent clearance of the drug from the patient's system. PK is used to ensure that there is sufficient concentration of the drug to maintain the required efficacy of the drug, while ensuring that the concentration levels remain below the toxic threshold. Pharmacokinetic (PK) models are often combined with Pharmacodynamic (PD) models, which model the positive effects of the drug, such as the  binding of a drug to the biological target, and/or undesirable side effects, to form a full PKPD model of the drug-body interaction. This example will only focus on PK, neglecting the interaction with a PD model.

![Fig 1](https://sabs-r3.github.io/software-engineering-projects/fig/pk1.jpg)

PK enables the following processes to be quantified:

- Absorption
- Distribution
- Metabolism
- Excretion

These are often referred to as ADME, and taken together describe the drug concentration in the body when medicine is prescribed. These ADME processes are typically described by zeroth-order or first-order *rate* reactions modelling the dynamics of the quantity of drug $q$, with a given rate parameter $k$, for example:

\\[
\frac{dq}{dt} = -k^*,
\\]

\\[
\frac{dq}{dt} = -k q.
\\]

The body itself is modelled as one or more *compartments*, each of which is defined as a kinetically homogeneous unit (these compartments do not relate to specific organs in the body, unlike Physiologically based pharmacokinetic, PBPK, modeling). There is typically a main *central* compartment into which the drug is administered and from which the drug is excreted from the body, combined with zero or more *peripheral* compartments to which the drug can be distributed to/from the central compartment (See Fig 2). Each 
peripheral compartment is only connected to the central compartment.

![Fig 2](https://sabs-r3.github.io/software-engineering-projects/fig/pk2.svg)

The following example PK model describes the two-compartment model shown diagrammatically in the figure above. The time-dependent variables to be solved are the drug quantity in the central and peripheral compartments, $q_c$ and $q_{p1}$ (units: [ng]) respectively.

\\[
\frac{dq_c}{dt} = \text{Dose}(t) - \frac{q_c}{V_c} CL - Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ),
\\]

\\[
\frac{dq_{p1}}{dt} =  Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ).
\\]

This model describes an *intravenous bolus* dosing protocol, with a linear clearance from the central compartment (non-linear clearance processes are also possible, but not considered here). The dose function $\text{Dose}(t)$ will consist of instantaneous doses of $X$ ng of the drug at one or more time points. The other input parameters to the model are:
- \\(V_c\\) [mL], the volume of the central compartment
- \\(V_{p1}\\) [mL], the volume of the first peripheral compartment
- \\(CL\\) [mL/h], the clearance/elimination rate from the central compartment
- \\(Q_{p1}\\) [mL/h], the transition rate between central compartment and peripheral compartment 1

We will solve this system of ODEs using the DiffSol crate. Rather than trying to write down the dose function as a mathematical function, we will neglect the dose function from the equations and instead using DiffSol's API to specify the dose at specific time points. 

First lets write down the equations in the standard form of a first order ODE system:

\\[
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)
\\]

where

\\[
\mathbf{y} = \begin{bmatrix} q_c \\\\ q_{p1} \end{bmatrix}
\\]

and

\\[
\mathbf{f}(\mathbf{y}, t) = \begin{bmatrix} - \frac{q_c}{V_c} CL - Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ) \\\\ Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ) \end{bmatrix}
\\]

