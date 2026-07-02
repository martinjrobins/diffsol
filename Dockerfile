FROM ubuntu:24.04
COPY ode_solvers_ci /ode_solvers_ci
CMD ["/ode_solvers_ci"]
