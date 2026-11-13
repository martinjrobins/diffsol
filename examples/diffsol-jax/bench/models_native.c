// The benchmark models as native C callbacks of diffsol-c: the baseline for the cost of the
// callback interface, with no PJRT. They obey the lane layout of `ode_callbacks.rs` (views
// `[n, P, B]`, strides in bytes, broadcast inputs have stride 0).
//
// - Robertson (3 states, 3 parameters), stiff;
// - Lotka-Volterra (2 states, 4 parameters), not stiff;
// - N coupled oscillators (2N states, 3 parameters), not stiff.
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "diffsol.h"

// Element i of lane (k, b) of a view.
static inline double get(const TensorView* v, size_t i, size_t k, size_t b) {
  return *(const double*)(v->ptr + i * v->strides[0] + k * v->strides[1] + b * v->strides[2]);
}
static inline double* at(TensorViewMut* v, size_t i, size_t k, size_t b) {
  return (double*)(v->ptr + i * v->strides[0] + k * v->strides[1] + b * v->strides[2]);
}

#define FOR_LANES(y)                         \
  for (size_t b = 0; b < (y)->shape[2]; b++) \
    for (size_t k = 0; k < (y)->shape[1]; k++)

// ---- Robertson: dx = -k1 x + k3 y z, dy = k1 x - k2 y^2 - k3 y z, dz = k2 y^2 ----------------

static int32_t rhs(const TensorView* x, const TensorView* p, double t, TensorViewMut* y,
                   void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double k1 = get(p, 0, k, b), k2 = get(p, 1, k, b), k3 = get(p, 2, k, b);
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b), x2 = get(x, 2, k, b);
    *at(y, 0, k, b) = -k1 * x0 + k3 * x1 * x2;
    *at(y, 1, k, b) = k1 * x0 - k2 * x1 * x1 - k3 * x1 * x2;
    *at(y, 2, k, b) = k2 * x1 * x1;
  }
  return 0;
}

// y = J v
static int32_t rhs_jac(const TensorView* x, const TensorView* p, double t, const TensorView* v,
                       TensorViewMut* y, void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double k1 = get(p, 0, k, b), k2 = get(p, 1, k, b), k3 = get(p, 2, k, b);
    double x1 = get(x, 1, k, b), x2 = get(x, 2, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b), v2 = get(v, 2, k, b);
    *at(y, 0, k, b) = -k1 * v0 + k3 * x2 * v1 + k3 * x1 * v2;
    *at(y, 1, k, b) = k1 * v0 - (2 * k2 * x1 + k3 * x2) * v1 - k3 * x1 * v2;
    *at(y, 2, k, b) = 2 * k2 * x1 * v1;
  }
  return 0;
}

// y = -J^T v
static int32_t rhs_adjoint(const TensorView* x, const TensorView* p, double t,
                           const TensorView* v, TensorViewMut* y, void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double k1 = get(p, 0, k, b), k2 = get(p, 1, k, b), k3 = get(p, 2, k, b);
    double x1 = get(x, 1, k, b), x2 = get(x, 2, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b), v2 = get(v, 2, k, b);
    *at(y, 0, k, b) = -(-k1 * v0 + k1 * v1);
    *at(y, 1, k, b) = -(k3 * x2 * v0 - (2 * k2 * x1 + k3 * x2) * v1 + 2 * k2 * x1 * v2);
    *at(y, 2, k, b) = -(k3 * x1 * v0 - k3 * x1 * v1);
  }
  return 0;
}

// y = -J_p^T v
static int32_t rhs_sens_adjoint(const TensorView* x, const TensorView* p, double t,
                                const TensorView* v, TensorViewMut* y, void* ud) {
  (void)p, (void)t, (void)ud;
  FOR_LANES(y) {
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b), x2 = get(x, 2, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b), v2 = get(v, 2, k, b);
    *at(y, 0, k, b) = -(-x0 * v0 + x0 * v1);
    *at(y, 1, k, b) = -(-x1 * x1 * v1 + x1 * x1 * v2);
    *at(y, 2, k, b) = -(x1 * x2 * v0 - x1 * x2 * v1);
  }
  return 0;
}

static int32_t init(const TensorView* p, double t, TensorViewMut* y, void* ud) {
  (void)p, (void)t, (void)ud;
  FOR_LANES(y) {
    *at(y, 0, k, b) = 1.0;
    *at(y, 1, k, b) = 0.0;
    *at(y, 2, k, b) = 0.0;
  }
  return 0;
}

static int32_t zero_mul(const TensorView* p, double t, const TensorView* v, TensorViewMut* y,
                        void* ud) {
  (void)p, (void)t, (void)v, (void)ud;
  FOR_LANES(y) for (size_t i = 0; i < y->shape[0]; i++) *at(y, i, k, b) = 0.0;
  return 0;
}

// ---- Robertson --------------------------------------------------------------------------------

// Fill the callbacks of the Robertson model (3 states, 3 parameters).
void robertson_desc(DiffsolOdeCallbacksF64* d) {
  *d = (DiffsolOdeCallbacksF64){0};
  d->nstates = 3;
  d->nparams = 3;
  d->rhs = rhs;
  d->rhs_jac = rhs_jac;
  d->rhs_adjoint = rhs_adjoint;
  d->rhs_sens_adjoint = rhs_sens_adjoint;
  d->init = init;
  d->init_sens = zero_mul;
  d->init_sens_adjoint = zero_mul;
}

size_t desc_size(void) { return sizeof(DiffsolOdeCallbacksF64); }

// ---- Lotka-Volterra: dx = a x - b x y, dy = c x y - d y, x(0) = 10, y(0) = 5 -------------------

static int32_t lv_rhs(const TensorView* x, const TensorView* p, double t, TensorViewMut* y,
                      void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double a = get(p, 0, k, b), bb = get(p, 1, k, b), c = get(p, 2, k, b), d = get(p, 3, k, b);
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b);
    *at(y, 0, k, b) = a * x0 - bb * x0 * x1;
    *at(y, 1, k, b) = c * x0 * x1 - d * x1;
  }
  return 0;
}

// y = J v, J = [[a - b y, -b x], [c y, c x - d]]
static int32_t lv_rhs_jac(const TensorView* x, const TensorView* p, double t, const TensorView* v,
                          TensorViewMut* y, void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double a = get(p, 0, k, b), bb = get(p, 1, k, b), c = get(p, 2, k, b), d = get(p, 3, k, b);
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b);
    *at(y, 0, k, b) = (a - bb * x1) * v0 - bb * x0 * v1;
    *at(y, 1, k, b) = c * x1 * v0 + (c * x0 - d) * v1;
  }
  return 0;
}

// y = -J^T v
static int32_t lv_rhs_adjoint(const TensorView* x, const TensorView* p, double t,
                              const TensorView* v, TensorViewMut* y, void* ud) {
  (void)t, (void)ud;
  FOR_LANES(y) {
    double a = get(p, 0, k, b), bb = get(p, 1, k, b), c = get(p, 2, k, b), d = get(p, 3, k, b);
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b);
    *at(y, 0, k, b) = -((a - bb * x1) * v0 + c * x1 * v1);
    *at(y, 1, k, b) = -(-bb * x0 * v0 + (c * x0 - d) * v1);
  }
  return 0;
}

// y = J_p v (v has 4 values)
static int32_t lv_rhs_sens(const TensorView* x, const TensorView* p, double t, const TensorView* v,
                           TensorViewMut* y, void* ud) {
  (void)p, (void)t, (void)ud;
  FOR_LANES(y) {
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b);
    *at(y, 0, k, b) = x0 * get(v, 0, k, b) - x0 * x1 * get(v, 1, k, b);
    *at(y, 1, k, b) = x0 * x1 * get(v, 2, k, b) - x1 * get(v, 3, k, b);
  }
  return 0;
}

// y = -J_p^T v (y has 4 values)
static int32_t lv_rhs_sens_adjoint(const TensorView* x, const TensorView* p, double t,
                                   const TensorView* v, TensorViewMut* y, void* ud) {
  (void)p, (void)t, (void)ud;
  FOR_LANES(y) {
    double x0 = get(x, 0, k, b), x1 = get(x, 1, k, b);
    double v0 = get(v, 0, k, b), v1 = get(v, 1, k, b);
    *at(y, 0, k, b) = -x0 * v0;
    *at(y, 1, k, b) = x0 * x1 * v0;
    *at(y, 2, k, b) = -x0 * x1 * v1;
    *at(y, 3, k, b) = x1 * v1;
  }
  return 0;
}

static int32_t lv_init(const TensorView* p, double t, TensorViewMut* y, void* ud) {
  (void)p, (void)t, (void)ud;
  FOR_LANES(y) {
    *at(y, 0, k, b) = 10.0;
    *at(y, 1, k, b) = 5.0;
  }
  return 0;
}

void lotka_volterra_desc(DiffsolOdeCallbacksF64* d) {
  *d = (DiffsolOdeCallbacksF64){0};
  d->nstates = 2;
  d->nparams = 4;
  d->rhs = lv_rhs;
  d->rhs_jac = lv_rhs_jac;
  d->rhs_sens = lv_rhs_sens;
  d->rhs_adjoint = lv_rhs_adjoint;
  d->rhs_sens_adjoint = lv_rhs_sens_adjoint;
  d->init = lv_init;
  d->init_sens = zero_mul;
  d->init_sens_adjoint = zero_mul;
}

// ---- N coupled oscillators ----------------------------------------------------------------------
// state [q_1..q_N, v_1..v_N], fixed ends q_0 = q_{N+1} = 0, parameters (k, beta, c):
//   dq_i = v_i,   dv_i = f(q_{i+1} - q_i) - f(q_i - q_{i-1}) - c v_i,   f(D) = k D + beta D^3
// q_i(0) = exp(-(i - (N + 1) / 2)^2), v_i(0) = 0. `user_data` points to N (a size_t).

static inline double qv(const TensorView* x, size_t n, long i, size_t k, size_t b) {
  // q_i with the fixed ends, i = 0..n+1 (1-based)
  return (i <= 0 || (size_t)i > n) ? 0.0 : get(x, (size_t)i - 1, k, b);
}

static int32_t osc_rhs(const TensorView* x, const TensorView* p, double t, TensorViewMut* y,
                       void* ud) {
  (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    double kk = get(p, 0, k, b), be = get(p, 1, k, b), c = get(p, 2, k, b);
    for (size_t i = 1; i <= n; i++) {
      double qi = qv(x, n, (long)i, k, b);
      double dp = qv(x, n, (long)i + 1, k, b) - qi, dm = qi - qv(x, n, (long)i - 1, k, b);
      double vi = get(x, n + i - 1, k, b);
      *at(y, i - 1, k, b) = vi;
      *at(y, n + i - 1, k, b) = kk * dp + be * dp * dp * dp - kk * dm - be * dm * dm * dm - c * vi;
    }
  }
  return 0;
}

// y = J u
static int32_t osc_rhs_jac(const TensorView* x, const TensorView* p, double t, const TensorView* u,
                           TensorViewMut* y, void* ud) {
  (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    double kk = get(p, 0, k, b), be = get(p, 1, k, b), c = get(p, 2, k, b);
    for (size_t i = 1; i <= n; i++) {
      double qi = qv(x, n, (long)i, k, b);
      double dp = qv(x, n, (long)i + 1, k, b) - qi, dm = qi - qv(x, n, (long)i - 1, k, b);
      double fp = kk + 3 * be * dp * dp, fm = kk + 3 * be * dm * dm;
      double ui = qv(u, n, (long)i, k, b);
      double up = qv(u, n, (long)i + 1, k, b) - ui, um = ui - qv(u, n, (long)i - 1, k, b);
      *at(y, i - 1, k, b) = get(u, n + i - 1, k, b);
      *at(y, n + i - 1, k, b) = fp * up - fm * um - c * get(u, n + i - 1, k, b);
    }
  }
  return 0;
}

// y = -J^T w
static int32_t osc_rhs_adjoint(const TensorView* x, const TensorView* p, double t,
                               const TensorView* w, TensorViewMut* y, void* ud) {
  (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    double kk = get(p, 0, k, b), be = get(p, 1, k, b), c = get(p, 2, k, b);
    for (size_t j = 1; j <= n; j++) {
      double qj = qv(x, n, (long)j, k, b);
      double dp = qv(x, n, (long)j + 1, k, b) - qj, dm = qj - qv(x, n, (long)j - 1, k, b);
      double fp = kk + 3 * be * dp * dp, fm = kk + 3 * be * dm * dm;
      double wv = get(w, n + j - 1, k, b);
      double wv_prev = j > 1 ? get(w, n + j - 2, k, b) : 0.0;
      double wv_next = j < n ? get(w, n + j, k, b) : 0.0;
      // (J^T w)_q_j = fm (w_v_{j-1} - w_v_j) + fp (w_v_{j+1} - w_v_j);  (J^T w)_v_j = w_q_j - c w_v_j
      *at(y, j - 1, k, b) = -(fm * (wv_prev - wv) + fp * (wv_next - wv));
      *at(y, n + j - 1, k, b) = -(get(w, j - 1, k, b) - c * wv);
    }
  }
  return 0;
}

// y = J_p u (u has 3 values)
static int32_t osc_rhs_sens(const TensorView* x, const TensorView* p, double t, const TensorView* u,
                            TensorViewMut* y, void* ud) {
  (void)p, (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    double uk = get(u, 0, k, b), ub = get(u, 1, k, b), uc = get(u, 2, k, b);
    for (size_t i = 1; i <= n; i++) {
      double qi = qv(x, n, (long)i, k, b);
      double dp = qv(x, n, (long)i + 1, k, b) - qi, dm = qi - qv(x, n, (long)i - 1, k, b);
      *at(y, i - 1, k, b) = 0.0;
      *at(y, n + i - 1, k, b) =
          (dp - dm) * uk + (dp * dp * dp - dm * dm * dm) * ub - get(x, n + i - 1, k, b) * uc;
    }
  }
  return 0;
}

// y = -J_p^T w (y has 3 values)
static int32_t osc_rhs_sens_adjoint(const TensorView* x, const TensorView* p, double t,
                                    const TensorView* w, TensorViewMut* y, void* ud) {
  (void)p, (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    double sk = 0, sb = 0, sc = 0;
    for (size_t i = 1; i <= n; i++) {
      double qi = qv(x, n, (long)i, k, b);
      double dp = qv(x, n, (long)i + 1, k, b) - qi, dm = qi - qv(x, n, (long)i - 1, k, b);
      double wv = get(w, n + i - 1, k, b);
      sk += wv * (dp - dm);
      sb += wv * (dp * dp * dp - dm * dm * dm);
      sc -= wv * get(x, n + i - 1, k, b);
    }
    *at(y, 0, k, b) = -sk;
    *at(y, 1, k, b) = -sb;
    *at(y, 2, k, b) = -sc;
  }
  return 0;
}

static int32_t osc_init(const TensorView* p, double t, TensorViewMut* y, void* ud) {
  (void)p, (void)t;
  size_t n = *(const size_t*)ud;
  FOR_LANES(y) {
    for (size_t i = 1; i <= n; i++) {
      double d = (double)i - (double)(n + 1) / 2.0;
      *at(y, i - 1, k, b) = exp(-d * d);
      *at(y, n + i - 1, k, b) = 0.0;
    }
  }
  return 0;
}

// `n` must stay valid until the ODE is freed.
void oscillators_desc(DiffsolOdeCallbacksF64* d, const size_t* n) {
  *d = (DiffsolOdeCallbacksF64){0};
  d->nstates = 2 * *n;
  d->nparams = 3;
  d->user_data = (void*)n;
  d->rhs = osc_rhs;
  d->rhs_jac = osc_rhs_jac;
  d->rhs_sens = osc_rhs_sens;
  d->rhs_adjoint = osc_rhs_adjoint;
  d->rhs_sens_adjoint = osc_rhs_sens_adjoint;
  d->init = osc_init;
  d->init_sens = zero_mul;
  d->init_sens_adjoint = zero_mul;
}
