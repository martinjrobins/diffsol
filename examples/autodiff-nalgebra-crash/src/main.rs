#![feature(autodiff)]

use std::autodiff::*;

use nalgebra::DVector;

fn logistic_rhs(x: &DVector<f64>, p: &DVector<f64>, y: &mut DVector<f64>) {
    y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
}

#[autodiff_forward(rhs_jvp, Dual, Const, Dual)]
#[autodiff_reverse(rhs_vjp, Duplicated, Const, Duplicated)]
#[autodiff_reverse(rhs_sens, Const, Duplicated, Duplicated)]
fn _autodiff_logistic(x: &DVector<f64>, p: &DVector<f64>, y: &mut DVector<f64>) {
    logistic_rhs(x, p, y)
}

fn main() {
    let x = DVector::from_vec(vec![0.5]);
    let p = DVector::from_vec(vec![1.0, 2.0]);
    let v = DVector::from_vec(vec![3.0]);
    let mut y = DVector::from_vec(vec![0.0]);
    let mut dy = DVector::from_vec(vec![0.0]);
    let mut dx = v.clone();
    let mut dp = DVector::from_vec(vec![1.0, 0.0]);

    // These calls force Enzyme to differentiate through DVector's internal
    // VecStorage pointer arithmetic, which crashes the Enzyme backend:
    //   "Cannot deduce adding type"
    //   UNREACHABLE executed at DiffeGradientUtils.cpp:524
    rhs_jvp(&x, &mut dx, &p, &mut y, &mut dy);
    println!("JVP: dy={}", dy[0]);

    rhs_vjp(&x, &mut dx, &p, &mut y, &mut dy);
    println!("VJP: dx={}", dx[0]);

    rhs_sens(&x, &p, &mut dp, &mut y, &mut dy);
    println!("sens: dp={:?}", dp.as_slice());
}
