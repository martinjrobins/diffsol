use diffsol::{
    Context, NalgebraContext, NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod, Vector,
    VectorCommon,
};

type M = NalgebraMat<f64>;
type V = NalgebraVec<f64>;

const NBATCH: usize = 3;

fn main() {
    // ANCHOR: context
    let context = NalgebraContext::with_nbatch(NBATCH);
    // ANCHOR_END: context

    // ANCHOR: builder
    let problem = OdeBuilder::<M>::new()
        .context(context)
        // Two parameters per lane. Parameter blocks are stored end to end:
        // [lane 0 p0, lane 0 p1, lane 1 p0, lane 1 p1, ...].
        .p([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
        .rhs(|y: &[f64], p: &[f64], _t: f64, dy: &mut [f64]| {
            // y, p, and dy are slices for one batch lane.
            dy[0] = p[0] * y[0] + p[1];
        })
        .init(|p: &[f64], _t: f64, y: &mut [f64]| y[0] = p[1], 1)
        .build()
        .unwrap();

    assert_eq!(problem.context().nbatch(), NBATCH);
    let mut solver = problem.tsit45().unwrap();
    solver.step().unwrap();
    // ANCHOR_END: builder

    // ANCHOR: vector_create
    let context = NalgebraContext::with_nbatch(NBATCH);
    let x = V::from_vec(vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0], context);
    // ANCHOR_END: vector_create
    //
    // ANCHOR: vector_inner
    assert_eq!(x.inner().nrows(), 2);
    assert_eq!(x.inner().ncols(), NBATCH);
    // ANCHOR_END: vector_inner

    // ANCHOR: vector_for_each
    let mut z = V::zeros(2, *x.context());
    z.for_each_batch([&x], |z, [x], _lane| {
        z[0] = x[0] + x[1];
        z[1] = x[0] - x[1];
    });
    // ANCHOR_END: vector_for_each

    // ANCHOR: vector_for_each_host
    let mut y = V::zeros(2, *x.context());
    y.for_each_batch_host([&x], |y, [x], lane| {
        y[0] = x[0] + lane as f64;
        y[1] = x[1] * 2.0;
    });
    // ANCHOR_END: vector_for_each_host

    // ANCHOR: vector_for_each_elem
    let mut scaled = V::zeros(2, *x.context());
    scaled.for_each_elem([&x], |value, [x], _lane, i| {
        *value = 10.0 * x[i];
    });
    // ANCHOR_END: vector_for_each_elem

    // ANCHOR: vector_reduce_elem
    let mut lane_max = V::zeros(1, *x.context());
    V::reduce_elem(&mut lane_max, [&x], 0.0, |[x], _lane, i| x[i], f64::max);
    assert_eq!(lane_max.clone_as_vec(), vec![2.0, 20.0, 200.0]);
    // ANCHOR_END: vector_reduce_elem

    // ANCHOR: vector_reduce_batch
    let mut batch_max = V::zeros(2, NalgebraContext::default());
    V::reduce_batch(&mut batch_max, [&x], 0.0, |[x], _lane, i| x[i], f64::max);
    assert_eq!(batch_max.clone_as_vec(), vec![100.0, 200.0]);
    // ANCHOR_END: vector_reduce_batch
}
