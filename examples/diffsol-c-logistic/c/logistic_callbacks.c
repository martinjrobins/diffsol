/*
 * Solve the logistic equation dy/dt = r y (1 - y), y(0) = 0.1, with equations that C callbacks
 * give. The program compares the solution with the exact solution and returns a non-zero exit
 * code if they do not agree.
 *
 * Build diffsol-c, then compile and link this file against the library and the generated
 * header in target/<profile>/include. From the root of the workspace:
 *
 *   cargo build -p diffsol-c --release
 *   cc examples/diffsol-c-logistic/c/logistic_callbacks.c -I target/release/include \
 *      -L target/release -ldiffsol_c -lm -Wl,-rpath,target/release -o logistic_callbacks
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "diffsol.h"

/* The data that each callback gets as `user_data`. */
typedef struct {
    int rhs_calls;
} UserData;

/* ANCHOR: callbacks */
/* The library calls each callback one time for all the lanes. Each vector is a view with the
 * shape [n, P, B]: element i of lane (k, b) is at ptr + i * strides[0] + k * strides[1]
 * + b * strides[2], in bytes. B is the number of lanes of the ODE (nbatch). P is 1, except in
 * sensitivity and adjoint solves. A stride can be 0: then the lanes on that axis share their
 * values. These two functions give the address of a value. */
static const double *at(const TensorView *v, size_t i, size_t k, size_t b) {
    return (const double *)(v->ptr + i * v->strides[0] + k * v->strides[1] + b * v->strides[2]);
}

static double *at_mut(TensorViewMut *v, size_t i, size_t k, size_t b) {
    return (double *)(v->ptr + i * v->strides[0] + k * v->strides[1] + b * v->strides[2]);
}

/* rhs: y = r x (1 - x). x and y have nstates values in each lane, p has nparams values. */
static int32_t rhs(const TensorView *x, const TensorView *p, double t, TensorViewMut *y,
                   void *user_data) {
    (void)t;
    ((UserData *)user_data)->rhs_calls += 1;
    for (size_t b = 0; b < y->shape[2]; b++) {
        for (size_t k = 0; k < y->shape[1]; k++) {
            double xi = *at(x, 0, k, b);
            *at_mut(y, 0, k, b) = *at(p, 0, k, b) * xi * (1.0 - xi);
        }
    }
    return 0; /* 0 is success; a different value tells the solver that the callback failed */
}

/* rhs_jac: y = J v, with J = dF/dx = r (1 - 2 x). The implicit solvers (for example BDF) need it. */
static int32_t rhs_jac(const TensorView *x, const TensorView *p, double t, const TensorView *v,
                       TensorViewMut *y, void *user_data) {
    (void)t;
    (void)user_data;
    for (size_t b = 0; b < y->shape[2]; b++) {
        for (size_t k = 0; k < y->shape[1]; k++) {
            *at_mut(y, 0, k, b) = *at(p, 0, k, b) * (1.0 - 2.0 * *at(x, 0, k, b)) * *at(v, 0, k, b);
        }
    }
    return 0;
}

/* init: the initial state y0(p, t). */
static int32_t init(const TensorView *p, double t, TensorViewMut *y, void *user_data) {
    (void)p;
    (void)t;
    (void)user_data;
    for (size_t b = 0; b < y->shape[2]; b++) {
        for (size_t k = 0; k < y->shape[1]; k++) {
            *at_mut(y, 0, k, b) = 0.1;
        }
    }
    return 0;
}
/* ANCHOR_END: callbacks */

static double exact(double r, double t) {
    const double y0 = 0.1;
    return y0 * exp(r * t) / (1.0 - y0 + y0 * exp(r * t));
}

static int fail(const char *what) {
    const char *message = diffsol_last_error_message();
    fprintf(stderr, "%s failed: %s\n", what, message ? message : "(no message)");
    return 1;
}

int main(void) {
    UserData data = {0};

    /* ANCHOR: create */
    /* Set all fields to zero first: a NULL callback means that it is not given. */
    DiffsolOdeCallbacksF64 callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.nstates = 1;
    callbacks.nparams = 1;
    callbacks.user_data = &data; /* data stays valid until the solver is freed */
    callbacks.free_user_data = NULL; /* the program owns data, so the library does not free it */
    callbacks.rhs = rhs;
    callbacks.rhs_jac = rhs_jac;
    callbacks.init = init;

    /* The last argument is nbatch, the number of lanes (copies of the system) to solve. */
    OdeWrapper *ode = diffsol_ode_new_callbacks_f64(&callbacks, DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE,
                                                    DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT,
                                                    DIFFSOL_ODE_SOLVER_TYPE_BDF, 1);
    if (ode == NULL) {
        return fail("diffsol_ode_new_callbacks_f64");
    }
    /* ANCHOR_END: create */
    diffsol_ode_set_rtol(ode, 1e-8);
    diffsol_ode_set_atol(ode, 1e-8);

    /* ANCHOR: solve */
    const double params[] = {2.0}; /* r */
    const double t_eval[] = {0.0, 0.5, 1.0, 2.0, 4.0};
    const size_t n_eval = sizeof(t_eval) / sizeof(t_eval[0]);

    /* params has 1 lane of 1 value. All the lanes of the ODE use a single lane of params. */
    SolutionWrapper *solution = NULL;
    if (diffsol_ode_solve_dense(ode, params, 1, 1, t_eval, n_eval, &solution) != DIFFSOL_OK) {
        diffsol_ode_free(ode);
        return fail("diffsol_ode_solve_dense");
    }

    /* ys is an (nout x n_eval x nbatch) array. Read it with its strides, which are in bytes. */
    Tensor *ys = NULL;
    if (diffsol_solution_wrapper_get_ys(solution, &ys) != DIFFSOL_OK) {
        diffsol_solution_wrapper_free(solution);
        diffsol_ode_free(ode);
        return fail("diffsol_solution_wrapper_get_ys");
    }
    const uint8_t *data_ptr = diffsol_tensor_ptr(ys);
    const size_t col_stride = diffsol_tensor_stride(ys, 1);

    int status = 0;
    printf("%8s %14s %14s\n", "t", "y", "exact");
    for (size_t j = 0; j < diffsol_tensor_dim(ys, 1); j++) {
        const double y = *(const double *)(data_ptr + j * col_stride);
        const double expected = exact(params[0], t_eval[j]);
        printf("%8.3f %14.8f %14.8f\n", t_eval[j], y, expected);
        if (fabs(y - expected) > 1e-5) {
            status = 1;
        }
    }
    printf("rhs was called %d times\n", data.rhs_calls);

    diffsol_tensor_free(ys);
    diffsol_solution_wrapper_free(solution);
    diffsol_ode_free(ode);
    /* ANCHOR_END: solve */
    if (status != 0) {
        fprintf(stderr, "the solution does not agree with the exact solution\n");
    }
    return status;
}
