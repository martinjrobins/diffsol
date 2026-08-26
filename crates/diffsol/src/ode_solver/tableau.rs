use crate::matrix::MAX_SMALL_COLS;
use crate::small::{SmallMat, SmallVec};
use crate::Scalar;

/// A matrix of a [`Tableau`] — `a`, or `beta`.
///
/// Capacity is [`MAX_SMALL_COLS`] stages squared, which fits `a` (`s x s`) and `beta`
/// (`s x poly_order`, with `poly_order <= s` for every method worth writing down) alike.
pub type TableauMat<T> = SmallMat<T, { MAX_SMALL_COLS * MAX_SMALL_COLS }>;

/// The `b`, `c` and `d` vectors of a [`Tableau`], `s` long.
pub type TableauVec<T> = SmallVec<T, MAX_SMALL_COLS>;

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
/// `a` and `beta` are stored *transposed*, because runge-kutta methods need a row of them.
/// The public API ([`new`](Tableau::new), [`a`](Tableau::a)) is in the natural orientation regardless.
#[derive(Clone, Copy, Debug)]
pub struct Tableau<T: Scalar> {
    /// `a` transposed: column `i` is stage `i`'s coefficient run.
    a_t: TableauMat<T>,
    b: TableauVec<T>,
    c: TableauVec<T>,
    d: TableauVec<T>,
    order: usize,
    /// `beta` transposed: column `i` is stage `i`'s polynomial coefficients.
    beta_t: Option<TableauMat<T>>,
}

impl<T: Scalar> Tableau<T> {
    /// TR-BDF2 method
    /// from R.E. Bank, W.M. Coughran Jr, W. Fichtner, E.H. Grosse, D.J. Rose and R.K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput.-Aided Design 4 (1985) 436-451.
    /// analysed in M.E. Hosea and L.F. Shampine. Analysis and implementation of TR-BDF2. Applied Numerical Mathematics, 20:21–37, 1996.
    ///
    /// continuous extension from :
    /// from Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv preprint arXiv:1803.01613.
    pub fn tr_bdf2() -> Self {
        let gamma = T::from_f64(2.0 - 2.0_f64.sqrt()).unwrap();
        let d = gamma / T::from_f64(2.0).unwrap();
        let w = T::from_f64(2.0_f64.sqrt() / 4.0).unwrap();

        let a = TableauMat::from_slice(
            3,
            3,
            &[T::zero(), d, w, T::zero(), d, w, T::zero(), T::zero(), d],
        );

        let b = TableauVec::from_slice(&[w, w, d]);
        let b_hat = [
            (T::one() - w) / T::from_f64(3.0).unwrap(),
            (T::from_f64(3.0).unwrap() * w + T::one()) / T::from_f64(3.0).unwrap(),
            d / T::from_f64(3.0).unwrap(),
        ];
        let mut d_vec = TableauVec::zeros(3);
        for (i, b_hat_i) in b_hat.iter().enumerate() {
            d_vec[i] = b[i] - *b_hat_i;
        }

        let beta = TableauMat::from_slice(
            3,
            2,
            &[
                T::from_f64(2.0).unwrap() * w,
                T::from_f64(2.0).unwrap() * w,
                gamma - T::one(),
                -w,
                -w,
                T::from_f64(2.0).unwrap() * w,
            ],
        );

        let c = TableauVec::from_slice(&[T::zero(), gamma, T::one()]);

        let order = 2;

        Self::new(a, b, c, d_vec, order, Some(beta))
    }

    /// A third order ESDIRK method
    /// from Jørgensen, J. B., Kristensen, M. R., & Thomsen, P. G. (2018). A family of ESDIRK integration methods. arXiv preprint arXiv:1803.01613.
    pub fn esdirk34() -> Self {
        let gamma = T::from_f64(0.435_866_521_508_459).unwrap();
        let a = TableauMat::from_slice(
            4,
            4,
            &[
                T::zero(),
                gamma,
                T::from_f64(0.140_737_774_724_706_2).unwrap(),
                T::from_f64(0.102_399_400_619_911).unwrap(),
                T::zero(),
                gamma,
                T::from_f64(-0.108_365_551_381_320_8).unwrap(),
                T::from_f64(-0.376_878_452_255_556_1).unwrap(),
                T::zero(),
                T::zero(),
                gamma,
                T::from_f64(0.838_612_530_127_186_1).unwrap(),
                T::zero(),
                T::zero(),
                T::zero(),
                gamma,
            ],
        );

        let b = TableauVec::from_slice(&[a[(3, 0)], a[(3, 1)], a[(3, 2)], a[(3, 3)]]);

        let c = TableauVec::from_slice(&[
            T::zero(),
            T::from_f64(0.871_733_043_016_918).unwrap(),
            T::from_f64(0.468_238_744_851_844_4).unwrap(),
            T::one(),
        ]);

        let d = TableauVec::from_slice(&[
            T::from_f64(-0.054_625_497_240_413_94).unwrap(),
            T::from_f64(-0.494_208_893_625_994_96).unwrap(),
            T::from_f64(0.221_934_499_735_064_66).unwrap(),
            T::from_f64(0.326_899_891_131_344_27).unwrap(),
        ]);

        Self::new(a, b, c, d, 3, None)
    }

    pub fn tsit45() -> Self {
        let c = TableauVec::from_slice(&[
            T::zero(),
            T::from_f64(0.161).unwrap(),
            T::from_f64(0.327).unwrap(),
            T::from_f64(0.9).unwrap(),
            T::from_f64(0.9800255409045097).unwrap(),
            T::one(),
            T::one(),
        ]);

        let b = TableauVec::from_slice(&[
            T::from_f64(0.09646076681806523).unwrap(),
            T::from_f64(0.01).unwrap(),
            T::from_f64(0.4798896504144996).unwrap(),
            T::from_f64(1.379008574103742).unwrap(),
            T::from_f64(-3.290069515436081).unwrap(),
            T::from_f64(2.324710524099774).unwrap(),
            T::zero(),
        ]);

        let d = TableauVec::from_slice(&[
            T::from_f64(-0.001_780_011_052_225_777).unwrap(),
            T::from_f64(-0.0008164344596567469).unwrap(),
            T::from_f64(0.007880878010261995).unwrap(),
            T::from_f64(-0.1447110071732629).unwrap(),
            T::from_f64(0.5823571654525552).unwrap(),
            T::from_f64(-0.45808210592918697).unwrap(),
            T::from_f64(0.015151515151515152).unwrap(),
        ]);

        // a matrix
        // [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[2] -  c[1], 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[3] -  c[1] - c[2], -6.359448489975075, 4.362295432869581, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[4] -  c[1] - c[2] - c[3], -11.74888356406283, 7.495539342889836, -0.09249506636175525, 0.0, 0.0, 0.0 ],
        // [ c[5] -  c[1] - c[2] - c[3] - c[4], -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.02826905039406838, 0.0, 0.0 ],
        // [ b[0], b[1], b[2], b[3], b[4], b[5], 0.0 ]
        let mut a = TableauMat::zeros(7, 7);
        a[(2, 1)] = T::from_f64(0.335_480_655_492_357).unwrap();
        a[(3, 1)] = T::from_f64(-6.359448489975075).unwrap();
        a[(4, 1)] = T::from_f64(-11.74888356406283).unwrap();
        a[(5, 1)] = T::from_f64(-12.92096931784711).unwrap();
        a[(3, 2)] = T::from_f64(4.362295432869581).unwrap();
        a[(4, 2)] = T::from_f64(7.495539342889836).unwrap();
        a[(5, 2)] = T::from_f64(8.159367898576159).unwrap();
        a[(4, 3)] = T::from_f64(-0.09249506636175525).unwrap();
        a[(5, 3)] = T::from_f64(-0.071_584_973_281_401).unwrap();
        a[(5, 4)] = T::from_f64(-0.02826905039406838).unwrap();
        for i in 1..7 {
            let mut a_sum = T::zero();
            for j in 1..i {
                a_sum += a[(i, j)];
            }
            a[(i, 0)] = c[i] - a_sum;
        }
        for j in 0..6 {
            a[(6, j)] = b[j];
        }

        // b0 = -1.05308849772902*t**4 + 2.91325546182191*t**3 - 2.76370619727483*t**2 + 1.0*t
        // b1 = 0.1017*t**4 - 0.2234*t**3 + 0.1317*t**2
        // b2 = 2.49062728565125*t**4 - 5.9410338721315*t**3 + 3.93029623689475*t**2
        // b3 = -16.5481028892449*t**4 + 30.3381886302823*t**3 - 12.4110771669337*t**2
        // b4 = 47.3795219628193*t**4 - 88.1789048947664*t**3 + 37.509313416511*t**2
        // b5 = -34.8706578614966*t**4 + 65.0918946747937*t**3 - 27.8965262891973*t**2
        // b6 = 2.5*t**4 - 4.0*t**3 + 1.5*t**2

        //r11 = convert(T, 1.0)

        //r12 = convert(T, -2.763706197274826)
        //r22 = convert(T, 0.13169999999999998)
        //r32 = convert(T, 3.9302962368947516)
        //r42 = convert(T, -12.411077166933676)
        //r52 = convert(T, 37.50931341651104)
        //r62 = convert(T, -27.896526289197286)
        //r72 = convert(T, 1.5)

        //r13 = convert(T, 2.9132554618219126)
        //r23 = convert(T, -0.2234)
        //r33 = convert(T, -5.941033872131505)
        //r43 = convert(T, 30.33818863028232)
        //r53 = convert(T, -88.1789048947664)
        //r63 = convert(T, 65.09189467479366)
        //r73 = convert(T, -4)

        //r14 = convert(T, -1.0530884977290216)
        //r24 = convert(T, 0.1017)
        //r34 = convert(T, 2.490627285651253)
        //r44 = convert(T, -16.548102889244902)
        //r54 = convert(T, 47.37952196281928)
        //r64 = convert(T, -34.87065786149661)
        //r74 = convert(T, 2.5)

        let beta = TableauMat::from_slice(
            7,
            4,
            &[
                T::one(),
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(),
                T::zero(),
                T::from_f64(-2.76370619727483).unwrap(),
                T::from_f64(0.1317).unwrap(),
                T::from_f64(3.93029623689475).unwrap(),
                T::from_f64(-12.4110771669337).unwrap(),
                T::from_f64(37.509313416511).unwrap(),
                T::from_f64(-27.8965262891973).unwrap(),
                T::from_f64(1.5).unwrap(),
                T::from_f64(2.91325546182191).unwrap(),
                T::from_f64(-0.2234).unwrap(),
                T::from_f64(-5.9410338721315).unwrap(),
                T::from_f64(30.3381886302823).unwrap(),
                T::from_f64(-88.1789048947664).unwrap(),
                T::from_f64(65.0918946747937).unwrap(),
                T::from_f64(-4.0).unwrap(),
                T::from_f64(-1.05308849772902).unwrap(),
                T::from_f64(0.1017).unwrap(),
                T::from_f64(2.49062728565125).unwrap(),
                T::from_f64(-16.5481028892449).unwrap(),
                T::from_f64(47.3795219628193).unwrap(),
                T::from_f64(-34.8706578614966).unwrap(),
                T::from_f64(2.5).unwrap(),
            ],
        );

        let order = 4;
        Self::new(a, b, c, d, order, Some(beta))
    }

    /// Builds a tableau from coefficients in the natural orientation: `a[i, j]` is the
    /// coefficient of stage `j` in stage `i`, and `beta[i, k]` the `k`th polynomial coefficient
    /// of stage `i`. Both are transposed once here so that the runs the kernels consume are
    /// contiguous.
    ///
    /// Panics if `c` is longer than [`MAX_SMALL_COLS`], or if the shapes disagree.
    pub fn new(
        a: TableauMat<T>,
        b: TableauVec<T>,
        c: TableauVec<T>,
        d: TableauVec<T>,
        order: usize,
        beta: Option<TableauMat<T>>,
    ) -> Self {
        let s = c.len();
        assert!(
            s <= MAX_SMALL_COLS,
            "Invalid tableau, at most {MAX_SMALL_COLS} stages are supported"
        );
        assert_eq!(a.ncols(), s, "Invalid number of rows in a, expected {s}");
        assert_eq!(a.nrows(), s, "Invalid number of columns in a, expected {s}",);
        assert_eq!(b.len(), s, "Invalid number of elements in b, expected {s}",);
        assert_eq!(d.len(), s, "Invalid number of elements in d, expected {s}",);
        if let Some(beta) = &beta {
            assert_eq!(
                beta.nrows(),
                s,
                "Invalid number of rows in beta, expected {s}",
            );
        }
        Self {
            a_t: a.transposed(),
            b,
            c,
            d,
            order,
            beta_t: beta.map(|beta| beta.transposed()),
        }
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn s(&self) -> usize {
        self.c.len()
    }

    /// `a[i, j]`, in the natural orientation.
    pub fn a(&self, i: usize, j: usize) -> T {
        self.a_t[(j, i)]
    }

    /// The coefficients stage `i` applies to the earlier stages, `a[i, 0..i]`, contiguous.
    pub fn stage_coeffs(&self, i: usize) -> &[T] {
        &self.a_t.as_col_slice(i)[..i]
    }

    pub fn b(&self) -> &TableauVec<T> {
        &self.b
    }

    pub fn c(&self) -> &TableauVec<T> {
        &self.c
    }

    pub fn d(&self) -> &TableauVec<T> {
        &self.d
    }

    /// The `beta` matrix *transposed*, `poly_order x s`: column `i` holds stage `i`'s polynomial
    /// coefficients contiguously. [`transposed`](SmallMat::transposed) recovers the natural
    /// orientation [`new`](Tableau::new) takes.
    pub fn beta_t(&self) -> Option<&TableauMat<T>> {
        self.beta_t.as_ref()
    }
}
