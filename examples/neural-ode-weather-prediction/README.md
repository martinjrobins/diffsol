# Neural ode weather prediction example

This is an example of using Diffsol to solve a neural ODE for weather prediction, based on the blog post by Sebastian Callh [Neural ODEs for Weather Prediction](https://sebastiancallh.github.io/post/neural-ode-weather-forecast/)

Please see the [Diffsol book](https://martinjrobins.github.io/diffsol/) for more information on this example.

## Pre-processing the data

Generate a python virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can then run the pre-processing script to process the data:

```bash
python examples/neural-ode-weather-prediction/src/data/pre-process.py
```

## Generating the ONNX models

You will need to generate the ONNX models for the neural network used in the neural ODE. You can do this by running the following command:

```bash
python examples/neural-ode-weather-prediction/src/model/model.py
```

## Running the example

You can then build and run the example using cargo:

```bash
cargo run --example neural-ode-weather-prediction --features onnx --release
```