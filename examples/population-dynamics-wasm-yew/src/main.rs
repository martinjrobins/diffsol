use diffsol::{MatrixCommon, NalgebraVec, OdeBuilder, OdeEquations, OdeSolverMethod, Op, Vector};
use web_sys::HtmlInputElement;
use yew::prelude::*;
use yew_plotly::{
    plotly::{common::Mode, layout::Axis, Layout, Plot, Scatter},
    Plotly,
};
type M = diffsol::NalgebraMat<f64>;

#[function_component]
fn App() -> Html {
    let problem = use_mut_ref(|| {
        OdeBuilder::<M>::new()
            .rhs(|x, p, _t, y| {
                y[0] = p[0] * x[0] - p[1] * x[0] * x[1];
                y[1] = p[2] * x[0] * x[1] - p[3] * x[1];
            })
            .init(|_p, _t, y| y.fill(1.0), 2)
            .p(vec![2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0])
            .build()
            .unwrap()
    });
    let params = use_state(|| {
        NalgebraVec::from_vec(
            vec![2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0],
            *problem.borrow().eqn.context(),
        )
    });

    let onchange = |i: usize| {
        let params = params.clone();
        let problem = problem.clone();
        Callback::from(move |e: InputEvent| {
            let input: HtmlInputElement = e.target_unchecked_into();
            let value = input.value().parse::<f64>().unwrap();
            let mut new_params = NalgebraVec::clone(&params);
            new_params[i] = value;
            let mut problem = problem.borrow_mut();
            problem.eqn.set_params(&new_params);
            params.set(new_params);
        })
    };
    let oninput_a: Callback<InputEvent> = onchange(0);
    let oninput_b: Callback<InputEvent> = onchange(1);

    let (ys, ts) = {
        let problem = problem.borrow();
        let mut solver = problem.tsit45().unwrap();
        solver.solve(40.0).unwrap()
    };

    let prey: Vec<_> = ys.inner().row(0).into_iter().copied().collect();
    let predator: Vec<_> = ys.inner().row(1).into_iter().copied().collect();
    let time: Vec<_> = ts.into_iter().collect();

    let prey = Scatter::new(time.clone(), prey)
        .mode(Mode::Lines)
        .name("Prey");
    let predator = Scatter::new(time, predator)
        .mode(Mode::Lines)
        .name("Predator");

    let mut plot = Plot::new();
    plot.add_trace(prey);
    plot.add_trace(predator);

    let layout = Layout::new()
        .x_axis(Axis::new().title("t".into()))
        .y_axis(Axis::new().title("population".into()));
    plot.set_layout(layout);

    let a_str = format!("{}", params.get_index(0));
    let b_str = format!("{}", params.get_index(1));

    html! {
        <div>
            <h1>{ "Population Dynamics: Prey-Predator Model" }</h1>
            <Plotly plot={plot}/>
            <ul>
                <li>
                    <input oninput={oninput_a} type="range" id="a" name="a" min="0.1" max="3" step="0.1" value={a_str} />
                    <label for="a">{"a"}</label>
                </li>
                <li>
                    <input oninput={oninput_b} type="range" id="b" name="b" min="0.1" max="3" step="0.1" value={b_str} />
                    <label for="b">{"b"}</label>
                </li>
            </ul>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
