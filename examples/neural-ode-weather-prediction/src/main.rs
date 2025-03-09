use std::collections::{hash_map, HashMap};

use diffsol::{NalgebraLU, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSensAdjoint, OdeBuilder, OdeEquations, OdeEquationsRef, OdeSolverProblem, Op, UnitCallable};
use nalgebra::{DMatrix, DVector};
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::error::Result;
use ndarray::{Array1, ViewRepr};

type V = DVector<f64>;
type M = DMatrix<f64>;
type LS = NalgebraLU<f64>;

const BASE_MODEL_DIR: &str = "examples/neural-ode-weather-prediction/src/model/";
const BASE_DATA_DIR: &str = "examples/neural-ode-weather-prediction/src/data/";

struct NeuralOde {
    rhs: Session,
    rhs_jac_mul: Session,
    rhs_jac_transpose_mul: Session,
    rhs_sens_transpose_mul: Session,
    inputs: HashMap<String, Array1<f32>>,
    output: Array1<f32>,
    y0: V,
}

impl NeuralOde {
    fn new_session(filename: &str) -> Result<Session> {
        let full_filename = format!("{}{}", BASE_MODEL_DIR, filename);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(full_filename.as_str())?;
        Ok(session)
    }
    fn new(y0: Array1<f32>) -> Result<Self> {
        let rhs = Self::new_session("rhs.onnx")?;
        let rhs_jac_mul = Self::new_session("rhs_jac_mul.onnx")?;
        let rhs_jac_transpose_mul = Self::new_session("rhs_jac_transpose_mul.onnx")?;
        let rhs_sens_transpose_mul = Self::new_session("rhs_sens_transpose_mul.onnx")?;
        let nparams = 2338;
        let mut rng = rand::thread_rng();
        let elem = Uniform::new(0.0, 1.0);
        let params = Array1::from_shape_fn((nparams,), |_| elem.sample(&mut rng));
        let inputs = inputs! {
            "p" => params,
            "y" => y0.clone(),
            "v" => y0.clone(),
        }?;

        Ok(Self {
            y0,
            rhs,
            rhs_jac_mul,
            rhs_jac_transpose_mul,
            rhs_sens_transpose_mul,
            inputs,
            output,
        })
    }
    
    fn data_dim(&self) -> usize {
        y0.len()
    }
}

impl Op for NeuralOde {
    type T = f64;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        self.data_dim()
    }
    fn nout(&self) -> usize {
        self.data_dim()
    }
    fn nparams(&self) -> usize {
        self.inputs["p"].len()
    }
}

impl<'a> OdeEquationsRef<'a> for NeuralOde {
    type Mass = ();
    type Rhs = Rhs;
    type Root = ();
    type Init = Init;
    type Out = ();
}

impl OdeEquations for NeuralOde {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        Rhs(self)
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        None
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        Init(self)
    }

    fn set_params(&mut self, p: &Self::V) {
        todo!()
    }

    fn get_params(&self, p: &mut Self::V) {
        todo!()
    }
}

struct Init<'a>(&'a NeuralOde);

impl Op for Init {
    type M = M;
    type V = V;
    type T = f64;
    fn nout(&self) -> usize {
        2
    }
    fn nparams(&self) -> usize {
        2338
    }
    fn nstates(&self) -> usize {
        2
    }
}

impl NonLinearOp for Init {
    fn call_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.0.y0);
    }
}

impl NonLinearOpSensAdjoint for Init {
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

struct Rhs<'a>(&'a NeuralOde);

impl Op for Rhs {
    type M = M;
    type V = V;
    type T = f64;
    fn nout(&self) -> usize {
        2
    }
    fn nparams(&self) -> usize {
        2338
    }
    fn nstates(&self) -> usize {
        2
    }
}

impl NonLinearOp for Rhs {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.inputs["y"].assign(x);
        self.0.rhs.run(self.inputs).unwrap();
        let y_data = self.rhs.output["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y[0] = y_data[0] as f64;
        y[1] = y_data[1] as f64;
    }
}

impl NonLinearOpJacobian for Rhs {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.inputs["y"].assign(x);
        self.0.inputs["v"].assign(v);
        self.0.rhs_jac_mul.run(self.inputs).unwrap();
        let y_data = self.rhs_jac_mul.output["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y[0] = y_data[0] as f64;
        y[1] = y_data[1] as f64;
    }
}

impl NonLinearOpAdjoint for Rhs {
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V) {
        self.0.inputs["y"].assign(x);
        self.0.inputs["v"].assign(v);
        self.0.rhs_jac_transpose_mul.run(self.inputs).unwrap();
        let y_data = self.rhs_jac_transpose_mul.output["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y[0] = y_data[0] as f64;
        y[1] = y_data[1] as f64;
    }
}

impl NonLinearOpSensAdjoint for Rhs {
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V) {
        self.0.inputs["y"].assign(x);
        self.0.inputs["v"].assign(v);
        self.0.rhs_sens_transpose_mul.run(self.inputs).unwrap();
        let y_data
            = self.rhs_sens_transpose_mul.output["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y[0] = y_data[0] as f64;
        y[1] = y_data[1] as f64;
    }
}

struct AdamW {
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    m: f64,
    m_hat: f64,
    v: f64,
    v_hat: f64,
    lambda: f64,
}

impl AdamW {
    fn new() -> Self {
        let lr = 1e-3;
        let betas = (0.9, 0.999);
        let eps = 1e-8;
        let m = 0.0;
        let m_hat = 0.0;
        let v = 0.0;
        let v_hat = 0.0;
        let lambda = 1e-2;
        Self {
            lr,
            betas,
            eps,
            m,
            m_hat,
            v,
            v_hat,
            lambda,
        }
    }

    fn step(&mut self, params: &mut V, grads: &V) {
        params *= 1.0 - self.lr * self.lambda;
        self.m = self.betas.0 * self.m + (1.0 - self.betas.0) * grads;
        self.v = self.betas.1 * self.v + (1.0 - self.betas.1) * grads.component_mul(&grads);
        self.m_hat = self.m / (1.0 - self.betas.0);
        self.v_hat = self.v / (1.0 - self.betas.1);
        params -= self.lr * self.m_hat / (self.v_hat.sqrt() + self.eps);
    }
}

fn loss_fn(problem: &mut OdeSolverProblem<NeuralOde>, p: &V, ts_data: &[f64], ys_data: &M, g_m: &mut M) -> Error<V> {
    let ts_data = data.column(0).as_slice().unwrap();
    problem.set_param(p);
    let solver = problem.bdf<LS>();
    let (c, ys) = match solver.solve_dense_with_checkpointing(&ts_data, None) {
        Ok(ys) => ys,
        Err(e) => return Err(Error::new("Failed to solve ODE: {}", e)),
    };
    for j in 0..g_m.ncols() {
        let g_m_i = 2.0 * (ys.column(j) - ys_data.column(j));
        g_m.column_mut(j).copy_from(&g_m_i);
    }
    let adjoint_solver = problem.bdf_solver_adjoint::<LS, _>(c, Some(1)).unwrap();
    match adjoint_solver.solve_adjoint_backwards_pass(ts_data, &[g_m]) {
        Ok(soln) => soln.into_common().sg[0],
        Err(e) => Err(Error::new("Failed to solve adjoint ODE: {}", e)),
    }
}

fn predict_fn(problem: &mut OdeSolverProblem<NeuralOde>, p: &V, ts_data: &[f64], ys_data: &M) -> Result<M> {
    let ts_data = data.column(0).as_slice().unwrap();
    problem.set_param(p);
    let solver = problem.bdf<LS>();
    let ys = solver.solve_dense(&ts_data, None)?;
    Ok(ys)
}
    

fn train_one_round(problem: &mut OdeSolverProblem<NeuralOde>, data: &Array2<f64>, p: &mut V) {
    let ts_data = data.column(0).to_owned();
    let ys_data = data.slice(s![1.., ..]).to_owned();
    let gm = M::zeros(problem.nout(), ts_data.len());
    let adam = AdamW::new();
    for _ in 0..1000 {
        match loss_fn(problem, p, ts_data, ys_data, g_m) {
            Ok(g) => adam.step(p, g),
            Err(e) => {
                panic!("{}", e);
            }
        };
    }
}
    

fn main() -> Result<()> {
    // load data
    let file = File::open(format!("{}{}", BASE_DATA_DIR, "MonthlyDelhiClimate.csv"))?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data: Array2<f64> = reader.deserialize_array2_dynamic()?;
    
    // y0 is first row of data
    let y0 = data.row(0).to_owned();

    let problem = OdeBuilder::<M>::new()
        .build_from_eqn(NeuralOde::new(y0))
        .unwrap();
    
    
    let mut p = V::zeros(problem.nparams());
    problem.get_params(&mut p);
    
    let ntimes = data.nrows() - 1;
    // train 4 data points at a time
    for i in 0..ntimes / 4 {
        // train
        let data = data.slice(s![i..i+4, ..]).to_owned();
        train_one_round(&mut problem, &data, &mut p);
        
        // predict
        
    }
    Ok(())
}