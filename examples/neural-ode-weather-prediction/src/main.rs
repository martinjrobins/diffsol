use std::cell::RefCell;
use std::fs::{self, File};
use std::ops::MulAssign;

use csv::ReaderBuilder;
use diffsol::{
    error::DiffsolError, AdjointOdeSolverMethod, DenseMatrix, Matrix, MatrixCommon,
    NalgebraContext, NalgebraLU, NalgebraMat, NalgebraVec, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSensAdjoint, OdeBuilder, OdeEquations, OdeEquationsRef,
    OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op, Scale, UnitCallable, Vector,
    VectorCommon, VectorView, VectorViewMut,
};
use diffsol::{ConstantOp, ConstantOpSensAdjoint};
use ndarray::Array1;
use ort::error::Result;
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use plotly::common::{DashType, Line, Mode};
use plotly::layout::{Axis, GridPattern, LayoutGrid};
use plotly::{Layout, Plot, Scatter};
use rand::distr::{Distribution, Uniform};

type T = f64;
type V = NalgebraVec<T>;
type M = NalgebraMat<T>;
type C = NalgebraContext;
type LS = NalgebraLU<T>;

const BASE_MODEL_DIR: &str = "examples/neural-ode-weather-prediction/src/model/";
const BASE_DATA_DIR: &str = "examples/neural-ode-weather-prediction/src/data/";
const BASE_OUTPUT_DIR: &str = "examples/neural-ode-weather-prediction/";

struct NeuralOde {
    rhs: Session,
    rhs_jac_mul: Session,
    rhs_jac_transpose_mul: Session,
    rhs_sens_transpose_mul: Session,
    input_y: RefCell<Array1<f32>>,
    input_v: RefCell<Array1<f32>>,
    input_p: Array1<f32>,
    y0: V,
}

impl NeuralOde {
    fn new_session(filename: &str) -> Result<Session> {
        let full_filename = format!("{BASE_MODEL_DIR}{filename}");
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(full_filename.as_str())?;
        Ok(session)
    }
    fn new(y0: V) -> Result<Self> {
        let rhs = Self::new_session("rhs.onnx")?;
        let rhs_jac_mul = Self::new_session("rhs_jac_mul.onnx")?;
        let rhs_jac_transpose_mul = Self::new_session("rhs_jac_transpose_mul.onnx")?;
        let rhs_sens_transpose_mul = Self::new_session("rhs_sens_transpose_mul.onnx")?;
        let mut nparams = 0;
        for input in rhs.inputs.iter() {
            if input.name == "p" {
                nparams = input.input_type.tensor_dimensions().unwrap()[0] as usize;
                break;
            }
        }
        let mut rng = rand::rng();
        let elem = Uniform::<f32>::new(0.0, 1.0).unwrap();
        let params = Array1::from_shape_fn((nparams,), |_| elem.sample(&mut rng));
        let y0_ndarray = Array1::from_shape_fn((y0.len(),), |i| y0[i] as f32);

        Ok(Self {
            y0,
            rhs,
            rhs_jac_mul,
            rhs_jac_transpose_mul,
            rhs_sens_transpose_mul,
            input_p: params,
            input_v: RefCell::new(y0_ndarray.clone()),
            input_y: RefCell::new(y0_ndarray),
        })
    }

    fn data_dim(&self) -> usize {
        self.y0.len()
    }
}

impl Op for NeuralOde {
    type T = T;
    type V = V;
    type M = M;
    type C = C;
    fn nstates(&self) -> usize {
        self.data_dim()
    }
    fn nout(&self) -> usize {
        self.data_dim()
    }
    fn nparams(&self) -> usize {
        self.input_p.len()
    }
    fn context(&self) -> &Self::C {
        self.y0.context()
    }
}

impl<'a> OdeEquationsRef<'a> for NeuralOde {
    type Mass = UnitCallable<M>;
    type Rhs = Rhs<'a>;
    type Root = UnitCallable<M>;
    type Init = Init<'a>;
    type Out = UnitCallable<M>;
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
        self.input_p
            .iter_mut()
            .zip(p.inner().iter())
            .for_each(|(input_p, p)| *input_p = *p as f32);
    }

    fn get_params(&self, p: &mut Self::V) {
        p.inner_mut()
            .iter_mut()
            .zip(self.input_p.iter())
            .for_each(|(p, input_p)| *p = *input_p as f64);
    }
}

struct Init<'a>(&'a NeuralOde);

impl Op for Init<'_> {
    type M = M;
    type V = V;
    type T = T;
    type C = C;
    fn nout(&self) -> usize {
        self.0.nout()
    }
    fn nparams(&self) -> usize {
        self.0.nparams()
    }
    fn nstates(&self) -> usize {
        self.0.nstates()
    }
    fn context(&self) -> &Self::C {
        self.0.context()
    }
}

impl ConstantOp for Init<'_> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.0.y0);
    }
}

impl ConstantOpSensAdjoint for Init<'_> {
    fn sens_transpose_mul_inplace(&self, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

struct Rhs<'a>(&'a NeuralOde);

impl Op for Rhs<'_> {
    type M = M;
    type V = V;
    type T = T;
    type C = C;
    fn nout(&self) -> usize {
        self.0.nout()
    }
    fn nparams(&self) -> usize {
        self.0.nparams()
    }
    fn nstates(&self) -> usize {
        self.0.nstates()
    }
    fn context(&self) -> &Self::C {
        self.0.context()
    }
}

impl NonLinearOp for Rhs<'_> {
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
        let mut y_input = self.0.input_y.borrow_mut();
        y_input
            .iter_mut()
            .zip(x.inner().iter())
            .for_each(|(y, x)| *y = *x as f32);
        let outputs = self
            .0
            .rhs
            .run(
                inputs![
                    "p" => self.0.input_p.view(),
                    "y" => y_input.view(),
                ]
                .unwrap(),
            )
            .unwrap();
        let y_data = outputs["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y.inner_mut()
            .iter_mut()
            .zip(y_data.as_slice().unwrap())
            .for_each(|(y, x)| *y = *x as f64);
    }
}

impl NonLinearOpJacobian for Rhs<'_> {
    fn jac_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut y_input = self.0.input_y.borrow_mut();
        y_input
            .iter_mut()
            .zip(x.inner().iter())
            .for_each(|(y, x)| *y = *x as f32);
        let mut v_input = self.0.input_v.borrow_mut();
        v_input
            .iter_mut()
            .zip(v.inner().iter())
            .for_each(|(v, x)| *v = *x as f32);
        let outputs = self
            .0
            .rhs_jac_mul
            .run(
                inputs![
                    "y" => y_input.view(),
                    "v" => v_input.view(),
                    "p" => self.0.input_p.view(),
                ]
                .unwrap(),
            )
            .unwrap();
        let y_data = outputs["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y.inner_mut()
            .iter_mut()
            .zip(y_data.as_slice().unwrap())
            .for_each(|(y, x)| *y = *x as f64);
    }
}

impl NonLinearOpAdjoint for Rhs<'_> {
    fn jac_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut y_input = self.0.input_y.borrow_mut();
        y_input
            .iter_mut()
            .zip(x.inner().iter())
            .for_each(|(y, x)| *y = *x as f32);
        let mut v_input = self.0.input_v.borrow_mut();
        v_input
            .iter_mut()
            .zip(v.inner().iter())
            .for_each(|(v, x)| *v = *x as f32);
        let outputs = self
            .0
            .rhs_jac_transpose_mul
            .run(
                inputs![
                    "y" => y_input.view(),
                    "v" => v_input.view(),
                    "p" => self.0.input_p.view(),
                ]
                .unwrap(),
            )
            .unwrap();
        let y_data = outputs["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y.inner_mut()
            .iter_mut()
            .zip(y_data.as_slice().unwrap())
            .for_each(|(y, x)| *y = *x as f64);
    }
}

impl NonLinearOpSensAdjoint for Rhs<'_> {
    fn sens_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut y_input = self.0.input_y.borrow_mut();
        y_input
            .iter_mut()
            .zip(x.inner().iter())
            .for_each(|(y, x)| *y = *x as f32);
        let mut v_input = self.0.input_v.borrow_mut();
        v_input
            .iter_mut()
            .zip(v.inner().iter())
            .for_each(|(v, x)| *v = *x as f32);
        let outputs = self
            .0
            .rhs_sens_transpose_mul
            .run(
                inputs![
                    "y" => y_input.view(),
                    "v" => v_input.view(),
                    "p" => self.0.input_p.view(),
                ]
                .unwrap(),
            )
            .unwrap();
        let y_data = outputs["Identity_1:0"].try_extract_tensor::<f32>().unwrap();
        y.inner_mut()
            .iter_mut()
            .zip(y_data.as_slice().unwrap())
            .for_each(|(y, x)| *y = *x as f64);
    }
}

struct AdamW {
    lr: T,
    betas: (T, T),
    eps: T,
    m: V,
    m_hat: V,
    v: V,
    v_hat: V,
    grads2: V,
    lambda: T,
    t: i32,
}

impl AdamW {
    fn new(nparams: usize, ctx: C) -> Self {
        let lr = 1e-2;
        let betas = (0.9, 0.999);
        let eps = 1e-8;
        let m = V::zeros(nparams, ctx);
        let m_hat = V::zeros(nparams, ctx);
        let v = V::zeros(nparams, ctx);
        let v_hat = V::zeros(nparams, ctx);
        let grads2 = V::zeros(nparams, ctx);
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
            grads2,
            t: 0,
        }
    }

    fn step(&mut self, params: &mut V, grads: &V) {
        self.t += 1;
        params.mul_assign(Scale(1.0 - self.lr * self.lambda));
        self.m.axpy(1.0 - self.betas.0, grads, self.betas.0);
        self.grads2.copy_from(grads);
        self.grads2.component_mul_assign(grads);
        self.v.axpy(1.0 - self.betas.1, &self.grads2, self.betas.1);
        self.m_hat = &self.m * Scale(1.0 / (1.0 - self.betas.0.powi(self.t)));
        self.v_hat = &self.v * Scale(1.0 / (1.0 - self.betas.1.powi(self.t)));
        params
            .inner_mut()
            .iter_mut()
            .zip(self.v_hat.inner().iter())
            .zip(self.m_hat.inner().iter())
            .for_each(|((params_i, v_hat_i), m_hat_i)| {
                *params_i -= self.lr * m_hat_i / (v_hat_i.sqrt() + self.eps)
            });
    }
}

fn loss_fn(
    problem: &mut OdeSolverProblem<NeuralOde>,
    p: &V,
    ts_data: &[T],
    ys_data: &M,
    g_m: &mut M,
) -> Result<(T, V), DiffsolError> {
    problem.eqn.set_params(p);
    let (c, ys) = problem
        .bdf::<LS>()?
        .solve_dense_with_checkpointing(ts_data, None)?;
    let mut loss = 0.0;
    for j in 0..g_m.ncols() {
        let delta = ys.column(j) - ys_data.column(j);
        loss += delta.inner().dot(delta.inner());
        let g_m_i = delta * Scale(2.0);
        g_m.column_mut(j).copy_from(&g_m_i);
    }
    let adjoint_solver = problem.bdf_solver_adjoint::<LS, _>(c, Some(1)).unwrap();
    let soln = adjoint_solver.solve_adjoint_backwards_pass(ts_data, &[g_m])?;
    Ok((loss, soln.into_common().sg.pop().unwrap()))
}

fn predict_fn(
    problem: &mut OdeSolverProblem<NeuralOde>,
    p: &V,
    ts_data: &[T],
) -> Result<M, DiffsolError> {
    problem.eqn.set_params(p);
    problem.bdf::<LS>()?.solve_dense(ts_data)
}

fn train_one_round(
    problem: &mut OdeSolverProblem<NeuralOde>,
    ts_data: &[T],
    ys_data: &M,
    p: &mut V,
) {
    let mut gm = M::zeros(problem.eqn.nout(), ts_data.len(), *problem.context());
    let mut adam = AdamW::new(problem.eqn.nparams(), *problem.context());
    for _ in 0..150 {
        match loss_fn(problem, p, ts_data, ys_data, &mut gm) {
            Ok((loss, g)) => {
                println!("loss: {loss}");
                adam.step(p, &g)
            }
            Err(e) => {
                panic!("{}", e);
            }
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // load data
    let ctx = NalgebraContext;
    let file = File::open(format!("{BASE_DATA_DIR}MonthlyDelhiClimate.csv"))?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let nrows = reader.records().count();
    let file = File::open(format!("{BASE_DATA_DIR}MonthlyDelhiClimate.csv"))?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let data_dim = 4;
    let mut ys_data = M::zeros(data_dim, nrows, ctx);
    let mut ts_data = vec![0.0; nrows];
    for (j, row) in reader.records().enumerate() {
        let row = row?;
        for (i, val) in row.iter().enumerate() {
            if i == 0 {
                ts_data[j] = val.parse::<T>()?;
            } else {
                ys_data[(i - 1, j)] = val.parse::<T>()?;
            }
        }
    }

    // y0 is first row of data
    let y0 = ys_data.column(0).into_owned();
    let eqn = NeuralOde::new(y0)?;
    let nparams = eqn.nparams();
    let mut rng = rand::rng();
    let elem = Uniform::<f64>::new(-0.1, 0.1).unwrap();
    let p = (0..nparams)
        .map(|_| elem.sample(&mut rng))
        .collect::<Vec<_>>();
    let mut problem = OdeBuilder::<M>::new().p(p).build_from_eqn(eqn)?;

    let mut p = V::zeros(problem.eqn.nparams(), *problem.context());
    problem.eqn.get_params(&mut p);

    let ntimes = ts_data.len();
    for i in 1..ntimes / 4 {
        // train
        let batch_len = i * 4;
        let ts_data_batch = &ts_data[0..batch_len];
        let mut ys_data_batch = M::zeros(ys_data.nrows(), batch_len, *problem.context());
        for j in 0..batch_len {
            ys_data_batch
                .column_mut(j)
                .copy_from_view(&ys_data.column(j));
        }
        println!("Training on data points: {batch_len}");
        train_one_round(&mut problem, ts_data_batch, &ys_data_batch, &mut p);

        // predict
        let predict_len = ntimes;
        let predict_ys = predict_fn(&mut problem, &p, &ts_data[0..predict_len])?;

        let ts_data_predict = ts_data[batch_len..predict_len].to_vec();

        // plot
        let mut plot = Plot::new();
        let mut mean_temp_data = Vec::with_capacity(predict_len);
        let mut mean_temp_train = Vec::with_capacity(batch_len);
        let mut mean_temp_predict = Vec::with_capacity(predict_len - batch_len);
        let mut humidity_data = Vec::with_capacity(predict_len);
        let mut humidity_train = Vec::with_capacity(batch_len);
        let mut humidity_predict = Vec::with_capacity(predict_len - batch_len);
        let mut wind_speed_data = Vec::with_capacity(predict_len);
        let mut wind_speed_train = Vec::with_capacity(batch_len);
        let mut wind_speed_predict = Vec::with_capacity(predict_len - batch_len);
        let mut meanpressure_data = Vec::with_capacity(predict_len);
        let mut meanpressure_train = Vec::with_capacity(batch_len);
        let mut meanpressure_predict = Vec::with_capacity(predict_len - batch_len);
        for j in 0..predict_len {
            let data = ys_data.column(j);
            mean_temp_data.push(data[0]);
            humidity_data.push(data[1]);
            wind_speed_data.push(data[2]);
            meanpressure_data.push(data[3]);
            let predict = predict_ys.column(j);
            if j < batch_len {
                mean_temp_train.push(predict[0]);
                humidity_train.push(predict[1]);
                wind_speed_train.push(predict[2]);
                meanpressure_train.push(predict[3]);
            } else {
                mean_temp_predict.push(predict[0]);
                humidity_predict.push(predict[1]);
                wind_speed_predict.push(predict[2]);
                meanpressure_predict.push(predict[3]);
            }
        }
        let mean_temp_data_plt = Scatter::new(ts_data.clone(), mean_temp_data)
            .name("mean_temp_data")
            .mode(Mode::Markers);
        let mean_temp_predict_plt = Scatter::new(ts_data_predict.clone(), mean_temp_predict)
            .name("mean_temp_predict")
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash));
        let mean_temp_train_plt = Scatter::new(ts_data_batch.to_vec(), mean_temp_train)
            .name("mean_temp_train")
            .mode(Mode::Lines);
        let humidity_data_plt = Scatter::new(ts_data.clone(), humidity_data)
            .name("humidity_data")
            .x_axis("x2")
            .y_axis("y2")
            .mode(Mode::Markers);
        let humidity_predict_plt = Scatter::new(ts_data_predict.clone(), humidity_predict)
            .name("humidity_predict")
            .x_axis("x2")
            .y_axis("y2")
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash));
        let humidity_train_plt = Scatter::new(ts_data_batch.to_vec(), humidity_train)
            .name("humidity_train")
            .x_axis("x2")
            .y_axis("y2")
            .mode(Mode::Lines);
        let wind_speed_data_plt = Scatter::new(ts_data.clone(), wind_speed_data)
            .name("wind_speed_data")
            .x_axis("x3")
            .y_axis("y3")
            .mode(Mode::Markers);
        let wind_speed_predict_plt = Scatter::new(ts_data_predict.clone(), wind_speed_predict)
            .name("wind_speed_predict")
            .x_axis("x3")
            .y_axis("y3")
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash));
        let wind_speed_train_plt = Scatter::new(ts_data_batch.to_vec(), wind_speed_train)
            .name("wind_speed_train")
            .x_axis("x3")
            .y_axis("y3")
            .mode(Mode::Lines);
        let meanpressure_data_plt = Scatter::new(ts_data.clone(), meanpressure_data)
            .name("meanpressure_data")
            .x_axis("x4")
            .y_axis("y4")
            .mode(Mode::Markers);
        let meanpressure_predict_plt = Scatter::new(ts_data_predict.clone(), meanpressure_predict)
            .name("meanpressure_predict")
            .x_axis("x4")
            .y_axis("y4")
            .mode(Mode::Lines)
            .line(Line::new().dash(DashType::Dash));
        let meanpressure_train_plt = Scatter::new(ts_data_batch.to_vec(), meanpressure_train)
            .name("meanpressure_train")
            .x_axis("x4")
            .y_axis("y4")
            .mode(Mode::Lines);
        plot.add_trace(mean_temp_data_plt);
        plot.add_trace(mean_temp_predict_plt);
        plot.add_trace(mean_temp_train_plt);
        plot.add_trace(humidity_data_plt);
        plot.add_trace(humidity_predict_plt);
        plot.add_trace(humidity_train_plt);
        plot.add_trace(wind_speed_data_plt);
        plot.add_trace(wind_speed_predict_plt);
        plot.add_trace(wind_speed_train_plt);
        plot.add_trace(meanpressure_data_plt);
        plot.add_trace(meanpressure_predict_plt);
        plot.add_trace(meanpressure_train_plt);

        let layout = Layout::new()
            .grid(
                LayoutGrid::new()
                    .rows(4)
                    .columns(1)
                    .pattern(GridPattern::Independent),
            )
            .y_axis(Axis::new().range(vec![-2., 2.]))
            .y_axis2(Axis::new().range(vec![-2., 2.]))
            .y_axis3(Axis::new().range(vec![-2., 2.]))
            .y_axis4(Axis::new().range(vec![-0.75, 0.5]));
        plot.set_layout(layout);
        let plot_html = plot.to_inline_html(Some("neural-ode-weather-prediction"));
        fs::write(
            format!("{BASE_OUTPUT_DIR}neural-ode-weather_{i}").as_str(),
            plot_html,
        )
        .expect("Unable to write file");
    }
    Ok(())
}
