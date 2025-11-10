use crate::{DenseMatrix, Vector};
use num_traits::{FromPrimitive, One, Zero};

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
        let gamma = M::T::from_f64(2.0 - 2.0_f64.sqrt()).unwrap();
        let d = gamma / M::T::from_f64(2.0).unwrap();
        let w = M::T::from_f64(2.0_f64.sqrt() / 4.0).unwrap();

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
                (M::T::one() - w) / M::T::from_f64(3.0).unwrap(),
                (M::T::from_f64(3.0).unwrap() * w + M::T::one()) / M::T::from_f64(3.0).unwrap(),
                d / M::T::from_f64(3.0).unwrap(),
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
                M::T::from_f64(2.0).unwrap() * w,
                M::T::from_f64(2.0).unwrap() * w,
                gamma - M::T::one(),
                -w,
                -w,
                M::T::from_f64(2.0).unwrap() * w,
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
        let gamma = M::T::from_f64(0.435_866_521_508_459).unwrap();
        let a = M::from_vec(
            4,
            4,
            vec![
                M::T::zero(),
                gamma,
                M::T::from_f64(0.140_737_774_724_706_2).unwrap(),
                M::T::from_f64(0.102_399_400_619_911).unwrap(),
                M::T::zero(),
                gamma,
                M::T::from_f64(-0.108_365_551_381_320_8).unwrap(),
                M::T::from_f64(-0.376_878_452_255_556_1).unwrap(),
                M::T::zero(),
                M::T::zero(),
                gamma,
                M::T::from_f64(0.838_612_530_127_186_1).unwrap(),
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
                M::T::from_f64(0.871_733_043_016_918).unwrap(),
                M::T::from_f64(0.468_238_744_851_844_4).unwrap(),
                M::T::one(),
            ],
            ctx.clone(),
        );

        let d = M::V::from_vec(
            vec![
                M::T::from_f64(-0.054_625_497_240_413_94).unwrap(),
                M::T::from_f64(-0.494_208_893_625_994_96).unwrap(),
                M::T::from_f64(0.221_934_499_735_064_66).unwrap(),
                M::T::from_f64(0.326_899_891_131_344_27).unwrap(),
            ],
            ctx.clone(),
        );

        Self::new(a, b, c, d, 3, None)
    }

    pub fn tsit45(ctx: M::C) -> Self {
        let c = M::V::from_vec(
            vec![
                M::T::zero(),
                M::T::from_f64(0.161).unwrap(),
                M::T::from_f64(0.327).unwrap(),
                M::T::from_f64(0.9).unwrap(),
                M::T::from_f64(0.9800255409045097).unwrap(),
                M::T::one(),
                M::T::one(),
            ],
            ctx.clone(),
        );

        let b = M::V::from_vec(
            vec![
                M::T::from_f64(0.09646076681806523).unwrap(),
                M::T::from_f64(0.01).unwrap(),
                M::T::from_f64(0.4798896504144996).unwrap(),
                M::T::from_f64(1.379008574103742).unwrap(),
                M::T::from_f64(-3.290069515436081).unwrap(),
                M::T::from_f64(2.324710524099774).unwrap(),
                M::T::zero(),
            ],
            ctx.clone(),
        );

        let d = M::V::from_vec(
            vec![
                M::T::from_f64(-0.001_780_011_052_225_777).unwrap(),
                M::T::from_f64(-0.0008164344596567469).unwrap(),
                M::T::from_f64(0.007880878010261995).unwrap(),
                M::T::from_f64(-0.1447110071732629).unwrap(),
                M::T::from_f64(0.5823571654525552).unwrap(),
                M::T::from_f64(-0.45808210592918697).unwrap(),
                M::T::from_f64(0.015151515151515152).unwrap(),
            ],
            ctx.clone(),
        );

        // a matrix
        // [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[2] -  c[1], 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[3] -  c[1] - c[2], -6.359448489975075, 4.362295432869581, 0.0, 0.0, 0.0, 0.0 ],
        // [ c[4] -  c[1] - c[2] - c[3], -11.74888356406283, 7.495539342889836, -0.09249506636175525, 0.0, 0.0, 0.0 ],
        // [ c[5] -  c[1] - c[2] - c[3] - c[4], -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.02826905039406838, 0.0, 0.0 ],
        // [ b[0], b[1], b[2], b[3], b[4], b[5], 0.0 ]
        let mut a = M::zeros(7, 7, ctx.clone());
        a.set_index(2, 1, M::T::from_f64(0.335_480_655_492_357).unwrap());
        a.set_index(3, 1, M::T::from_f64(-6.359448489975075).unwrap());
        a.set_index(4, 1, M::T::from_f64(-11.74888356406283).unwrap());
        a.set_index(5, 1, M::T::from_f64(-12.92096931784711).unwrap());
        a.set_index(3, 2, M::T::from_f64(4.362295432869581).unwrap());
        a.set_index(4, 2, M::T::from_f64(7.495539342889836).unwrap());
        a.set_index(5, 2, M::T::from_f64(8.159367898576159).unwrap());
        a.set_index(4, 3, M::T::from_f64(-0.09249506636175525).unwrap());
        a.set_index(5, 3, M::T::from_f64(-0.071_584_973_281_401).unwrap());
        a.set_index(5, 4, M::T::from_f64(-0.02826905039406838).unwrap());
        for i in 1..7 {
            let mut a_sum = M::T::zero();
            for j in 1..i {
                a_sum += a.get_index(i, j);
            }
            a.set_index(i, 0, c.get_index(i) - a_sum);
        }
        for j in 0..6 {
            a.set_index(6, j, b.get_index(j));
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

        let beta = M::from_vec(
            7,
            4,
            vec![
                M::T::one(),
                M::T::zero(),
                M::T::zero(),
                M::T::zero(),
                M::T::zero(),
                M::T::zero(),
                M::T::zero(),
                M::T::from_f64(-2.76370619727483).unwrap(),
                M::T::from_f64(0.1317).unwrap(),
                M::T::from_f64(3.93029623689475).unwrap(),
                M::T::from_f64(-12.4110771669337).unwrap(),
                M::T::from_f64(37.509313416511).unwrap(),
                M::T::from_f64(-27.8965262891973).unwrap(),
                M::T::from_f64(1.5).unwrap(),
                M::T::from_f64(2.91325546182191).unwrap(),
                M::T::from_f64(-0.2234).unwrap(),
                M::T::from_f64(-5.9410338721315).unwrap(),
                M::T::from_f64(30.3381886302823).unwrap(),
                M::T::from_f64(-88.1789048947664).unwrap(),
                M::T::from_f64(65.0918946747937).unwrap(),
                M::T::from_f64(-4.0).unwrap(),
                M::T::from_f64(-1.05308849772902).unwrap(),
                M::T::from_f64(0.1017).unwrap(),
                M::T::from_f64(2.49062728565125).unwrap(),
                M::T::from_f64(-16.5481028892449).unwrap(),
                M::T::from_f64(47.3795219628193).unwrap(),
                M::T::from_f64(-34.8706578614966).unwrap(),
                M::T::from_f64(2.5).unwrap(),
            ],
            ctx.clone(),
        );

        let order = 4;
        Self::new(a, b, c, d, order, Some(beta))
    }

    pub fn new(a: M, b: M::V, c: M::V, d: M::V, order: usize, beta: Option<M>) -> Self {
        let s = c.len();
        assert_eq!(a.ncols(), s, "Invalid number of rows in a, expected {s}");
        assert_eq!(a.nrows(), s, "Invalid number of columns in a, expected {s}",);
        assert_eq!(b.len(), s, "Invalid number of elements in b, expected {s}",);
        assert_eq!(c.len(), s, "Invalid number of elements in c, expected {s}",);
        if let Some(beta) = &beta {
            assert_eq!(
                beta.nrows(),
                s,
                "Invalid number of rows in beta, expected {s}",
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
