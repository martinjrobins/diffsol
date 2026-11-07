/*
 * Solve the logistic equation dy/dt = r y (1 - y), y(0) = 0.1, from DiffSL code that the
 * library compiles at run time (JIT). The program compares the solution with the exact solution
 * and returns a non-zero exit code if they do not agree.
 *
 * The library must have a JIT backend, for example the Cranelift backend:
 *
 *   cargo build -p diffsol-c --release --features diffsl-cranelift
 *   cc logistic_diffsl_jit.c -I target/release/include -L target/release -ldiffsol_c -lm \
 *      -Wl,-rpath,target/release -o logistic_diffsl_jit
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "diffsol.h"

#if !defined(DIFFSOL_DIFFSL_CRANELIFT) && !defined(DIFFSOL_DIFFSL_LLVM)
#error "diffsol-c was built without a JIT backend: enable diffsl-cranelift or diffsl-llvm"
#endif

/* ANCHOR: code */
/* The model in DiffSL. `in_i` are the parameters, `u_i` the state, `F_i` the right-hand side
 * and `out_i` the output. */
static const char *CODE = "in_i { r = 1 }\n"
                          "u_i { y = 0.1 }\n"
                          "F_i { r * y * (1 - y) }\n"
                          "out_i { y }\n";
/* ANCHOR_END: code */

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
    /* ANCHOR: create */
    /* Use Cranelift if the library has it, otherwise LLVM. */
#if defined(DIFFSOL_DIFFSL_CRANELIFT)
    const int32_t backend = DIFFSOL_JIT_BACKEND_TYPE_CRANELIFT;
#else
    const int32_t backend = DIFFSOL_JIT_BACKEND_TYPE_LLVM;
#endif
    printf("JIT backend: %s\n", diffsol_jit_backend_type_name(backend));

    OdeWrapper *ode = diffsol_ode_new_jit(CODE, backend, DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE,
                                          DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT,
                                          DIFFSOL_ODE_SOLVER_TYPE_BDF);
    if (ode == NULL) {
        return fail("diffsol_ode_new_jit");
    }
    /* ANCHOR_END: create */
    diffsol_ode_set_rtol(ode, 1e-8);
    diffsol_ode_set_atol(ode, 1e-8);

    /* ANCHOR: solve */
    const double params[] = {2.0}; /* r */
    const double t_eval[] = {0.0, 0.5, 1.0, 2.0, 4.0};
    const size_t n_eval = sizeof(t_eval) / sizeof(t_eval[0]);

    SolutionWrapper *solution = NULL;
    if (diffsol_ode_solve_dense(ode, params, 1, t_eval, n_eval, &solution) != DIFFSOL_OK) {
        diffsol_ode_free(ode);
        return fail("diffsol_ode_solve_dense");
    }

    /* ys is an (nout x n_eval) array. Read it with its strides, which are in bytes. */
    HostArray *ys = NULL;
    if (diffsol_solution_wrapper_get_ys(solution, &ys) != DIFFSOL_OK) {
        diffsol_solution_wrapper_free(solution);
        diffsol_ode_free(ode);
        return fail("diffsol_solution_wrapper_get_ys");
    }
    const uint8_t *data_ptr = diffsol_host_array_ptr(ys);
    const size_t col_stride = diffsol_host_array_stride(ys, 1);

    int status = 0;
    printf("%8s %14s %14s\n", "t", "y", "exact");
    for (size_t j = 0; j < diffsol_host_array_dim(ys, 1); j++) {
        const double y = *(const double *)(data_ptr + j * col_stride);
        const double expected = exact(params[0], t_eval[j]);
        printf("%8.3f %14.8f %14.8f\n", t_eval[j], y, expected);
        if (fabs(y - expected) > 1e-5) {
            status = 1;
        }
    }

    diffsol_host_array_free(ys);
    diffsol_solution_wrapper_free(solution);
    diffsol_ode_free(ode);
    /* ANCHOR_END: solve */
    if (status != 0) {
        fprintf(stderr, "the solution does not agree with the exact solution\n");
    }
    return status;
}
