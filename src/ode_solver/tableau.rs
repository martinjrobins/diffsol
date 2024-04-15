use crate::{DenseMatrix, Vector};
use num_traits::{One, Zero};

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
    pub fn tr_bdf2() -> Self {
        let gamma = M::T::from(2.0 - 2.0_f64.sqrt());
        let d = gamma / M::T::from(2.0);
        let w = M::T::from(2.0_f64.sqrt() / 4.0);

        let mut a = M::zeros(3, 3);
        a[(1, 0)] = d;
        a[(1, 1)] = d;

        a[(2, 0)] = w;
        a[(2, 1)] = w;
        a[(2, 2)] = d;

        let b = M::V::from_vec(vec![w, w, d]);
        let b_hat = M::V::from_vec(vec![
            (M::T::from(1.0) - w) / M::T::from(3.0),
            (M::T::from(3.0) * w + M::T::from(1.0)) / M::T::from(3.0),
            d / M::T::from(3.0),
        ]);
        let mut d = M::V::zeros(3);
        for i in 0..3 {
            d[i] = b[i] - b_hat[i];
        }

        let mut beta = M::zeros(3, 2);
        beta[(0, 0)] = M::T::from(2.0) * w;
        beta[(0, 1)] = -w;
        beta[(1, 0)] = M::T::from(2.0) * w;
        beta[(1, 1)] = -w;
        beta[(2, 0)] = gamma - M::T::from(1.0);
        beta[(2, 1)] = M::T::from(2.0) * w;

        let c = M::V::from_vec(vec![M::T::zero(), gamma, M::T::one()]);

        let order = 2;

        Self::new(a, b, c, d, order, Some(beta))
    }

    /// from Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv preprint arXiv:1803.01613.
    pub fn esdirk34() -> Self {
        let mut a = M::zeros(4, 4);
        let gamma = M::T::from(0.435_866_521_508_459);
        a[(1, 0)] = gamma;
        a[(1, 1)] = gamma;

        a[(2, 0)] = M::T::from(0.140_737_774_724_706_2);
        a[(2, 1)] = M::T::from(-0.108_365_551_381_320_8);
        a[(2, 2)] = gamma;

        a[(3, 0)] = M::T::from(0.102_399_400_619_911);
        a[(3, 1)] = M::T::from(-0.376_878_452_255_556_1);
        a[(3, 2)] = M::T::from(0.838_612_530_127_186_1);
        a[(3, 3)] = gamma;

        let b = M::V::from_vec(vec![a[(3, 0)], a[(3, 1)], a[(3, 2)], a[(3, 3)]]);

        let c = M::V::from_vec(vec![
            M::T::zero(),
            M::T::from(0.871_733_043_016_918),
            M::T::from(0.468_238_744_851_844_4),
            M::T::one(),
        ]);

        let d = M::V::from_vec(vec![
            M::T::from(-0.054_625_497_240_413_94),
            M::T::from(-0.494_208_893_625_994_96),
            M::T::from(0.221_934_499_735_064_66),
            M::T::from(0.326_899_891_131_344_27),
        ]);

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
