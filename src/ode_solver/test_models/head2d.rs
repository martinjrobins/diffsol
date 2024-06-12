//
//heatres: heat equation system residual function
//This uses 5-point central differencing on the interior points, and
//includes algebraic equations for the boundary values.
//So for each interior point, the residual component has the form
//   res_i = u'_i - (central difference)_i
//while for each boundary point, it is res_i = u_i.

use crate::{
    ode_solver::problem::OdeSolverSolution, Matrix, OdeBuilder, OdeEquations, OdeSolverProblem,
    Vector,
};
use num_traits::{One, Zero};

fn heat2d_rhs<M: Matrix, const MGRID: usize>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    // Initialize y to x, to take care of boundary equations.
    y.copy_from(x);
    let mm = M::T::from(MGRID as f64);

    let dx = M::T::one() / (mm - M::T::one());
    let coeff = M::T::one() / (dx * dx);

    // Loop over interior points; set y = (central difference).
    for j in 1..MGRID - 1 {
        let offset = MGRID * j;
        for i in 1..MGRID - 1 {
            let loc = offset + i;
            y[loc] = coeff
                * (x[loc - 1] + x[loc + 1] + x[loc - MGRID] + x[loc + MGRID]
                    - M::T::from(4.0) * x[loc]);
        }
    }
}

/* Jacobian matrix setup for MGRID>=4  */
fn heat2d_jacobian<M: Matrix, const MGRID: usize>() -> M {
    /* total num of nonzero elements */
    let total = 4 * MGRID + 8 * (MGRID - 2) + (MGRID - 4) * (MGRID + 4 * (MGRID - 2));
    let one = M::T::one();
    let mm = M::T::from(MGRID as f64);
    let four = M::T::from(4.0);

    let dx = M::T::one() / (mm - M::T::one());
    let beta = -four / (dx * dx);

    let mut colptrs = vec![0; MGRID * MGRID + 1];
    let mut rowvals = vec![0; total];
    let mut data = vec![M::T::zero(); total];
    //
    //-----------------------------------------------
    // set up number of elements in each column
    //-----------------------------------------------
    //
    //
    /**** first column block ****/
    colptrs[0] = 0;
    colptrs[1] = 1;
    /* count by twos in the middle  */
    for i in 2..MGRID {
        colptrs[i] = colptrs[i - 1] + 2;
    }
    colptrs[MGRID] = 2 * MGRID - 2;

    /**** second column block ****/
    colptrs[MGRID + 1] = 2 * MGRID;
    colptrs[MGRID + 2] = 2 * MGRID + 3;
    /* count by fours in the middle */
    for i in 0..MGRID - 4 {
        colptrs[MGRID + 3 + i] = colptrs[MGRID + 3 + i - 1] + 4;
    }
    colptrs[2 * MGRID - 1] = 2 * MGRID + 4 * (MGRID - 2) - 2;
    colptrs[2 * MGRID] = 2 * MGRID + 4 * (MGRID - 2);

    /**** repeated (MGRID-4 times) middle column blocks ****/
    let mut repeat = 0;
    for _i in 0..MGRID - 4 {
        colptrs[2 * MGRID + 1 + repeat] = colptrs[2 * MGRID + 1 + repeat - 1] + 2;
        colptrs[2 * MGRID + 1 + repeat + 1] = colptrs[2 * MGRID + 1 + repeat] + 4;

        /* count by fives in the middle */
        for j in 0..MGRID - 4 {
            colptrs[2 * MGRID + 1 + repeat + 2 + j] = colptrs[2 * MGRID + 1 + repeat + 1 + j] + 5;
        }
        colptrs[2 * MGRID + 1 + repeat + (MGRID - 4) + 2] =
            colptrs[2 * MGRID + 1 + repeat + (MGRID - 4) + 1] + 4;

        colptrs[2 * MGRID + 1 + repeat + (MGRID - 4) + 3] =
            colptrs[2 * MGRID + 1 + repeat + (MGRID - 4) + 2] + 2;

        repeat += MGRID; /* shift that accounts for accumulated number of columns */
    }

    /**** last-1 column block ****/
    colptrs[MGRID * MGRID - 2 * MGRID + 1] = total - 2 * MGRID - 4 * (MGRID - 2) + 2;
    colptrs[MGRID * MGRID - 2 * MGRID + 2] = total - 2 * MGRID - 4 * (MGRID - 2) + 5;
    for i in 0..MGRID - 4 {
        colptrs[MGRID * MGRID - 2 * MGRID + 3 + i] =
            colptrs[MGRID * MGRID - 2 * MGRID + 3 + i - 1] + 4;
    }
    colptrs[MGRID * MGRID - MGRID - 1] = total - 2 * MGRID;
    colptrs[MGRID * MGRID - MGRID] = total - 2 * MGRID + 2;

    /**** last column block ****/
    colptrs[MGRID * MGRID - MGRID + 1] = total - MGRID - (MGRID - 2) + 1;
    for i in 0..MGRID - 2 {
        colptrs[MGRID * MGRID - MGRID + 2 + i] = colptrs[MGRID * MGRID - MGRID + 2 + i - 1] + 2;
    }
    colptrs[MGRID * MGRID - 1] = total - 1;
    colptrs[MGRID * MGRID] = total;

    //
    //-----------------------------------------------
    // set up data stored
    //-----------------------------------------------
    //

    /**** first column block ****/
    data[0] = one;
    /* alternating pattern in data, separate loop for each pattern  */
    #[allow(clippy::needless_range_loop)]
    for i in 1..MGRID + (MGRID - 2) {
        data[i] = one;
    }
    for i in (2..MGRID + (MGRID - 2) - 1).step_by(2) {
        data[i] = -one / (dx * dx);
    }

    /**** second column block ****/
    data[MGRID + MGRID - 2] = one;
    data[MGRID + MGRID - 1] = -one / (dx * dx);
    data[MGRID + MGRID] = beta;
    data[MGRID + MGRID + 1] = -one / (dx * dx);
    data[MGRID + MGRID + 2] = -one / (dx * dx);

    /* middle data elements */
    for i in 0..(MGRID - 4) {
        data[MGRID + MGRID + 3 + 4 * i] = -one / (dx * dx);
    }
    for i in 0..(MGRID - 4) {
        data[MGRID + MGRID + 4 + 4 * i] = beta;
    }
    for i in 0..(MGRID - 4) {
        data[MGRID + MGRID + 5 + 4 * i] = -one / (dx * dx);
    }
    for i in 0..(MGRID - 4) {
        data[MGRID + MGRID + 6 + 4 * i] = -one / (dx * dx);
    }
    data[2 * MGRID + 4 * (MGRID - 2) - 5] = -one / (dx * dx);
    data[2 * MGRID + 4 * (MGRID - 2) - 4] = beta;
    data[2 * MGRID + 4 * (MGRID - 2) - 3] = -one / (dx * dx);
    data[2 * MGRID + 4 * (MGRID - 2) - 2] = -one / (dx * dx);
    data[2 * MGRID + 4 * (MGRID - 2) - 1] = one;

    /**** repeated (MGRID-4 times) middle column blocks ****/
    let mut repeat = 0;
    for _i in 0..MGRID - 4 {
        data[2 * MGRID + 4 * (MGRID - 2) + repeat] = one;
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + 1] = -one / (dx * dx);

        data[2 * MGRID + 4 * (MGRID - 2) + repeat + 2] = -one / (dx * dx);
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + 3] = beta;
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + 4] = -one / (dx * dx);
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + 5] = -one / (dx * dx);

        /* 5 in 5*j chosen since there are 5 elements in each column */
        /* this column loops MGRID-4 times within the outer loop */
        for j in 0..MGRID - 4 {
            data[2 * MGRID + 4 * (MGRID - 2) + repeat + 6 + 5 * j] = -one / (dx * dx);
            data[2 * MGRID + 4 * (MGRID - 2) + repeat + 7 + 5 * j] = -one / (dx * dx);
            data[2 * MGRID + 4 * (MGRID - 2) + repeat + 8 + 5 * j] = beta;
            data[2 * MGRID + 4 * (MGRID - 2) + repeat + 9 + 5 * j] = -one / (dx * dx);
            data[2 * MGRID + 4 * (MGRID - 2) + repeat + 10 + 5 * j] = -one / (dx * dx);
        }

        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 6] = -one / (dx * dx);
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 7] = -one / (dx * dx);
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 8] = beta;
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 9] = -one / (dx * dx);

        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 10] = -one / (dx * dx);
        data[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 11] = one;

        repeat += MGRID + 4 * (MGRID - 2); /* shift that accounts for accumulated columns and elements */
    }

    /**** last-1 column block ****/
    data[total - 6 * (MGRID - 2) - 4] = one;
    data[total - 6 * (MGRID - 2) - 3] = -one / (dx * dx);
    data[total - 6 * (MGRID - 2) - 2] = -one / (dx * dx);
    data[total - 6 * (MGRID - 2) - 1] = beta;
    data[total - 6 * (MGRID - 2)] = -one / (dx * dx);

    /* middle data elements */
    for i in 0..MGRID - 4 {
        data[total - 6 * (MGRID - 2) + 1 + 4 * i] = -one / (dx * dx);
    }
    for i in 0..MGRID - 4 {
        data[total - 6 * (MGRID - 2) + 2 + 4 * i] = -one / (dx * dx);
    }
    for i in 0..MGRID - 4 {
        data[total - 6 * (MGRID - 2) + 3 + 4 * i] = beta;
    }
    for i in 0..MGRID - 4 {
        data[total - 6 * (MGRID - 2) + 4 + 4 * i] = -one / (dx * dx);
    }
    data[total - 2 * (MGRID - 2) - 7] = -one / (dx * dx);
    data[total - 2 * (MGRID - 2) - 6] = -one / (dx * dx);
    data[total - 2 * (MGRID - 2) - 5] = beta;
    data[total - 2 * (MGRID - 2) - 4] = -one / (dx * dx);
    data[total - 2 * (MGRID - 2) - 3] = one;

    /**** last column block ****/
    data[total - 2 * (MGRID - 2) - 2] = one;
    /* alternating pattern in data, separate loop for each pattern  */
    for i in (total - 2 * (MGRID - 2) - 1..total - 2).step_by(2) {
        data[i] = -one / (dx * dx);
    }
    for i in (total - 2 * (MGRID - 2)..total - 1).step_by(2) {
        data[i] = one;
    }
    data[total - 1] = one;

    /*
     *-----------------------------------------------
     * row values
     *-----------------------------------------------
     */

    /**** first block ****/
    rowvals[0] = 0;
    /* alternating pattern in data, separate loop for each pattern */
    for i in (1..MGRID + (MGRID - 2)).step_by(2) {
        rowvals[i] = (i + 1) / 2;
    }
    for i in (2..MGRID + (MGRID - 2) - 1).step_by(2) {
        /* i+1 unnecessary here */
        rowvals[i] = i / 2 + MGRID;
    }

    /**** second column block ****/
    rowvals[MGRID + MGRID - 2] = MGRID;
    rowvals[MGRID + MGRID - 1] = MGRID + 1;
    rowvals[MGRID + MGRID] = MGRID + 1;
    rowvals[MGRID + MGRID + 1] = MGRID + 2;
    rowvals[MGRID + MGRID + 2] = 2 * MGRID + 1;

    /* middle row values */
    for i in 0..MGRID - 4 {
        rowvals[MGRID + MGRID + 3 + 4 * i] = MGRID + 1 + i;
    }
    for i in 0..MGRID - 4 {
        rowvals[MGRID + MGRID + 4 + 4 * i] = MGRID + 2 + i;
    }
    for i in 0..MGRID - 4 {
        rowvals[MGRID + MGRID + 5 + 4 * i] = MGRID + 3 + i;
    }
    for i in 0..MGRID - 4 {
        rowvals[MGRID + MGRID + 6 + 4 * i] = 2 * MGRID + 2 + i;
    }
    rowvals[2 * MGRID + 4 * (MGRID - 2) - 5] = MGRID + (MGRID - 2) - 1;
    rowvals[2 * MGRID + 4 * (MGRID - 2) - 4] = MGRID + (MGRID - 2); /* starting from here, add two diag patterns */
    rowvals[2 * MGRID + 4 * (MGRID - 2) - 3] = 2 * MGRID + (MGRID - 2);
    rowvals[2 * MGRID + 4 * (MGRID - 2) - 2] = MGRID + (MGRID - 2);
    rowvals[2 * MGRID + 4 * (MGRID - 2) - 1] = MGRID + (MGRID - 2) + 1;

    /**** repeated (MGRID-4 times) middle column blocks ****/
    let mut repeat = 0;
    for i in 0..MGRID - 4 {
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat] = MGRID + (MGRID - 2) + 2 + MGRID * i;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 1] = MGRID + (MGRID - 2) + 2 + MGRID * i + 1;

        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 2] =
            MGRID + (MGRID - 2) + 2 + MGRID * i + 1 - MGRID;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 3] = MGRID + (MGRID - 2) + 2 + MGRID * i + 1;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 4] = MGRID + (MGRID - 2) + 2 + MGRID * i + 2; /* *this */
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 5] =
            MGRID + (MGRID - 2) + 2 + MGRID * i + 1 + MGRID;

        /* 5 in 5*j chosen since there are 5 elements in each column */
        /* column repeats MGRID-4 times within the outer loop */
        for j in 0..MGRID - 4 {
            rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 6 + 5 * j] =
                MGRID + (MGRID - 2) + 2 + MGRID * i + 1 - MGRID + 1 + j;
            rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 7 + 5 * j] =
                MGRID + (MGRID - 2) + 2 + MGRID * i + 1 + j;
            rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 8 + 5 * j] =
                MGRID + (MGRID - 2) + 2 + MGRID * i + 2 + j;
            rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 9 + 5 * j] =
                MGRID + (MGRID - 2) + 2 + MGRID * i + 2 + 1 + j;
            rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + 10 + 5 * j] =
                MGRID + (MGRID - 2) + 2 + MGRID * i + 1 + MGRID + 1 + j;
        }

        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 6] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 7] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2 + MGRID - 1;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 8] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2 + MGRID; /* *this+MGRID */
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 9] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2 + 2 * MGRID;

        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 10] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2 + MGRID;
        rowvals[2 * MGRID + 4 * (MGRID - 2) + repeat + (MGRID - 4) * 5 + 11] =
            MGRID + (MGRID - 2) + 2 + MGRID * i - 2 + MGRID + 1;

        repeat += MGRID + 4 * (MGRID - 2); /* shift that accounts for accumulated columns and elements */
    }

    /**** last-1 column block ****/
    rowvals[total - 6 * (MGRID - 2) - 4] = MGRID * MGRID - 1 - 2 * (MGRID - 1) - 1;
    rowvals[total - 6 * (MGRID - 2) - 3] = MGRID * MGRID - 1 - 2 * (MGRID - 1); /* starting with this as base */
    rowvals[total - 6 * (MGRID - 2) - 2] = MGRID * MGRID - 1 - 2 * (MGRID - 1) - MGRID;
    rowvals[total - 6 * (MGRID - 2) - 1] = MGRID * MGRID - 1 - 2 * (MGRID - 1);
    rowvals[total - 6 * (MGRID - 2)] = MGRID * MGRID - 1 - 2 * (MGRID - 1) + 1;
    /* middle row values */
    for i in 0..MGRID - 4 {
        rowvals[total - 6 * (MGRID - 2) + 1 + 4 * i] =
            MGRID * MGRID - 1 - 2 * (MGRID - 1) - MGRID + 1 + i;
    }
    for i in 0..MGRID - 4 {
        rowvals[total - 6 * (MGRID - 2) + 2 + 4 * i] = MGRID * MGRID - 1 - 2 * (MGRID - 1) + i;
    }
    for i in 0..MGRID - 4 {
        rowvals[total - 6 * (MGRID - 2) + 3 + 4 * i] = MGRID * MGRID - 1 - 2 * (MGRID - 1) + 1 + i;
        /*copied above*/
    }
    for i in 0..MGRID - 4 {
        rowvals[total - 6 * (MGRID - 2) + 4 + 4 * i] = MGRID * MGRID - 1 - 2 * (MGRID - 1) + 2 + i;
    }
    rowvals[total - 2 * (MGRID - 2) - 7] = MGRID * MGRID - 2 * MGRID - 2;
    rowvals[total - 2 * (MGRID - 2) - 6] = MGRID * MGRID - MGRID - 3;
    rowvals[total - 2 * (MGRID - 2) - 5] = MGRID * MGRID - MGRID - 2;
    rowvals[total - 2 * (MGRID - 2) - 4] = MGRID * MGRID - MGRID - 2;
    rowvals[total - 2 * (MGRID - 2) - 3] = MGRID * MGRID - MGRID - 1;

    /* last column block */
    rowvals[total - 2 * (MGRID - 2) - 2] = MGRID * MGRID - MGRID;
    /* alternating pattern in data, separate loop for each pattern  */
    for i in 0..MGRID - 2 {
        rowvals[total - 2 * (MGRID - 2) - 1 + 2 * i] = MGRID * MGRID - 2 * MGRID + 1 + i;
    }
    for i in 0..MGRID - 2 {
        rowvals[total - 2 * (MGRID - 2) + 2 * i] = MGRID * MGRID - MGRID + 1 + i;
    }
    rowvals[total - 1] = MGRID * MGRID - 1;

    let mut triplets = Vec::with_capacity(total);
    for (j, cptr) in colptrs.windows(2).enumerate() {
        let start = cptr[0];
        let end = cptr[1];
        for i in start..end {
            triplets.push((rowvals[i], j, data[i]));
        }
    }

    M::try_from_triplets(MGRID * MGRID, MGRID * MGRID, triplets).unwrap()
}

fn heat2d_init<M: Matrix, const MGRID: usize>(_p: &M::V, _t: M::T) -> M::V {
    let mm = M::T::from(MGRID as f64);
    let mut uu = M::V::zeros(MGRID * MGRID);
    let bval = M::T::zero();
    let one = M::T::one();
    let dx = one / (mm - one);
    let mm1 = MGRID - 1;

    /* Initialize uu on all grid points. */
    for j in 0..MGRID {
        let yfact = dx * M::T::from(j as f64);
        let offset = MGRID * j;
        for i in 0..MGRID {
            let xfact = dx * M::T::from(i as f64);
            let loc = offset + i;
            uu[loc] = M::T::from(16.0) * xfact * (one - xfact) * yfact * (one - yfact);
        }
    }

    /* Finally, set values of u, up, and id at boundary points. */
    for j in 0..MGRID {
        let offset = MGRID * j;
        for i in 0..MGRID {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                uu[loc] = bval;
            }
        }
    }
    uu
}

fn heat2d_mass<M: Matrix, const MGRID: usize>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    beta: M::T,
    y: &mut M::V,
) {
    let mm = MGRID;
    let mm1 = mm - 1;
    for j in 0..mm {
        let offset = mm * j;
        for i in 0..mm {
            let loc = offset + i;
            if j == 0 || j == mm1 || i == 0 || i == mm1 {
                y[loc] *= beta;
            } else {
                y[loc] = x[loc] + beta * y[loc];
            }
        }
    }
}

pub fn head2d_problem<M: Matrix + 'static, const MGRID: usize>() -> (
    OdeSolverProblem<impl OdeEquations<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let jac = heat2d_jacobian::<M, MGRID>();
    let jac_mul = move |_x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V| {
        jac.gemv(M::T::one(), v, M::T::zero(), y);
    };
    let problem = OdeBuilder::new()
        .build_ode_with_mass(
            heat2d_rhs::<M, MGRID>,
            jac_mul,
            heat2d_mass::<M, MGRID>,
            heat2d_init::<M, MGRID>,
        )
        .unwrap();

    let mut soln = OdeSolverSolution::default();
    let data = vec![
        (vec![9.75461e-01], 0.0),
        (vec![8.24056e-01], 0.01),
        (vec![6.88097e-01], 0.02),
        (vec![4.70961e-01], 0.04),
        (vec![2.16312e-01], 0.08),
        (vec![4.53210e-02], 0.16),
        (vec![1.98864e-03], 0.32),
        (vec![3.83238e-06], 0.64),
        (vec![0.0], 1.28),
        (vec![0.0], 2.56),
        (vec![0.0], 5.12),
        (vec![0.0], 10.24),
    ];
    for (values, time) in data {
        soln.push(
            M::V::from_vec(values.into_iter().map(|v| v.into()).collect()),
            time.into(),
        );
    }
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use crate::LinearOp;

    use super::*;

    #[test]
    fn test_jacobian() {
        let jac = heat2d_jacobian::<nalgebra::DMatrix<f64>, 10>();
        insta::assert_yaml_snapshot!(jac.to_string());
    }

    #[test]
    fn test_mass() {
        let (problem, _soln) = head2d_problem::<nalgebra::DMatrix<f64>, 10>();
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        insta::assert_yaml_snapshot!(mass.to_string());
    }
}
