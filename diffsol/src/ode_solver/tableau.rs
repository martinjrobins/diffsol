use crate::{DenseMatrix, Vector};
use num_traits::{One, Zero};

/// A butcher tableau for a Runge-Kutta method.
///
/// The tableau is defined by the matrices `a`, `b`, `c` and `d` and the order of the method.
/// The butchers tableau is often depicted like this example of a 3-stage method:
///
/// ```text
/// c1 | a11 0   0
/// c2 | a21 a22 0
/// c3 | a31 a32 a33
/// -------------------
///   | b1  b2  b3  
///   | be1 be2 be3
/// -------------------
///   | d1  d2  d3
/// ```
///
/// where `be` is the embedded method for error control and `d` is the difference between the main and embedded method.
///
/// For continous extension methods, the beta matrix is also included.
///
#[derive(Clone)]
pub struct Tableau<M: DenseMatrix> {
    a: M,
    b: M::V,
    c: M::V,
    d: M::V,
    order: usize,
    beta: Option<M>,
}

impl<M: DenseMatrix> Tableau<M> {
    /// TR-BDF2 method
    /// from R.E. Bank, W.M. Coughran Jr, W. Fichtner, E.H. Grosse, D.J. Rose and R.K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput.-Aided Design 4 (1985) 436-451.
    /// analysed in M.E. Hosea and L.F. Shampine. Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20:21–37, 1996.
    ///
    /// continuous extension from :
    /// from Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv preprint arXiv:1803.01613.
    pub fn tr_bdf2(ctx: M::C) -> Self {
        let gamma = M::T::from(2.0 - 2.0_f64.sqrt());
        let d = gamma / M::T::from(2.0);
        let w = M::T::from(2.0_f64.sqrt() / 4.0);

        let a = M::from_vec(
            3,
            3,
            vec![
                M::T::zero(),
                d,
                w,
                M::T::zero(),
                d,
                w,
                M::T::zero(),
                M::T::zero(),
                d,
            ],
            ctx.clone(),
        );

        let b = M::V::from_vec(vec![w, w, d], ctx.clone());
        let b_hat = M::V::from_vec(
            vec![
                (M::T::from(1.0) - w) / M::T::from(3.0),
                (M::T::from(3.0) * w + M::T::from(1.0)) / M::T::from(3.0),
                d / M::T::from(3.0),
            ],
            ctx.clone(),
        );
        let mut d = M::V::zeros(3, ctx.clone());
        for i in 0..3 {
            d.set_index(i, b.get_index(i) - b_hat.get_index(i));
        }

        let beta = M::from_vec(
            3,
            2,
            vec![
                M::T::from(2.) * w,
                M::T::from(2.) * w,
                gamma - M::T::from(1.),
                -w,
                -w,
                M::T::from(2.) * w,
            ],
            ctx.clone(),
        );

        let c = M::V::from_vec(vec![M::T::zero(), gamma, M::T::one()], ctx.clone());

        let order = 2;

        Self::new(a, b, c, d, order, Some(beta))
    }

    /// A third order ESDIRK method
    /// from Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv preprint arXiv:1803.01613.
    pub fn esdirk34(ctx: M::C) -> Self {
        let gamma = M::T::from(0.435_866_521_508_459);
        let a = M::from_vec(
            4,
            4,
            vec![
                M::T::zero(),
                gamma,
                M::T::from(0.140_737_774_724_706_2),
                M::T::from(0.102_399_400_619_911),
                M::T::zero(),
                gamma,
                M::T::from(-0.108_365_551_381_320_8),
                M::T::from(-0.376_878_452_255_556_1),
                M::T::zero(),
                M::T::zero(),
                gamma,
                M::T::from(0.838_612_530_127_186_1),
                M::T::zero(),
                M::T::zero(),
                M::T::zero(),
                gamma,
            ],
            ctx.clone(),
        );

        let b = M::V::from_vec(
            vec![
                a.get_index(3, 0),
                a.get_index(3, 1),
                a.get_index(3, 2),
                a.get_index(3, 3),
            ],
            ctx.clone(),
        );

        let c = M::V::from_vec(
            vec![
                M::T::zero(),
                M::T::from(0.871_733_043_016_918),
                M::T::from(0.468_238_744_851_844_4),
                M::T::one(),
            ],
            ctx.clone(),
        );

        let d = M::V::from_vec(
            vec![
                M::T::from(-0.054_625_497_240_413_94),
                M::T::from(-0.494_208_893_625_994_96),
                M::T::from(0.221_934_499_735_064_66),
                M::T::from(0.326_899_891_131_344_27),
            ],
            ctx.clone(),
        );

        Self::new(a, b, c, d, 3, None)
    }

    pub fn new(a: M, b: M::V, c: M::V, d: M::V, order: usize, beta: Option<M>) -> Self {
        let s = c.len();
        assert_eq!(a.ncols(), s, "Invalid number of rows in a, expected {}", s);
        assert_eq!(
            a.nrows(),
            s,
            "Invalid number of columns in a, expected {}",
            s
        );
        assert_eq!(
            b.len(),
            s,
            "Invalid number of elements in b, expected {}",
            s
        );
        assert_eq!(
            c.len(),
            s,
            "Invalid number of elements in c, expected {}",
            s
        );
        if let Some(beta) = &beta {
            assert_eq!(
                beta.nrows(),
                s,
                "Invalid number of rows in beta, expected {}",
                s
            );
        }
        Self {
            a,
            b,
            c,
            d,
            order,
            beta,
        }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn s(&self) -> usize {
        self.c.len()
    }

    pub fn a(&self) -> &M {
        &self.a
    }

    pub fn b(&self) -> &M::V {
        &self.b
    }

    pub fn c(&self) -> &M::V {
        &self.c
    }

    pub fn d(&self) -> &M::V {
        &self.d
    }

    pub fn beta(&self) -> Option<&M> {
        self.beta.as_ref()
    }
}
