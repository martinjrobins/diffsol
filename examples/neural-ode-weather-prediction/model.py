import jax
import jax.numpy as jnp
import equinox as eqx
import functools as ft
import tensorflow as tf
from jax.experimental import jax2tf
from jax.flatten_util import ravel_pytree
import tf2onnx


def to_onnx(model, inputs, filename):
    sig = [tf.TensorSpec(inpt.shape, inpt.dtype) for inpt in inputs]
    inference_tf = jax2tf.convert(model, enable_xla=False)
    inference_tf = tf.function(inference_tf, autograph=False)
    inference_onnx = tf2onnx.convert.from_function(inference_tf, input_signature=sig)
    model_proto, _external_tensor_storage = inference_onnx
    with open(filename, "wb") as f:
        f.write(model_proto.SerializeToString())

data_dim = 2

class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, data_dim, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(data_dim, 64, key=key1),
            eqx.nn.Linear(64, 32, key=key2),
            eqx.nn.Linear(32, data_dim, key=key3),
        ]

    def __call__(self, x):
        x = jax.nn.silu(self.layers[0](x))  # Swish = SiLU
        x = jax.nn.silu(self.layers[1](x))
        x = self.layers[2](x)
        return x


key = jax.random.PRNGKey(0)
model = NeuralNetwork(data_dim=data_dim, key=key)
y = jnp.zeros((data_dim,))
v = jnp.zeros((data_dim,))
params, static = eqx.partition(model, eqx.is_array)
p, unravel_params = ravel_pytree(params)

def rhs(p, y):
    params = unravel_params(p)
    model = eqx.combine(params, static)
    return model(y)

def rhs_jac_mul(p, y, v):
    return jax.jvp(ft.partial(rhs, p), (y,), (v,))[1]


def rhs_jac_transpose_mul(p, y, v):
    return jax.vjp(ft.partial(rhs, p), y)[1](v)[0]


def rhs_sens_transpose_mul(p, y, v):
    return jax.vjp(ft.partial(rhs, y=y), p)[1](v)[0]


to_onnx(rhs, (p, y), "rhs.onnx")
to_onnx(rhs_jac_mul, (p, y, v), "rhs_jac_mul.onnx")
to_onnx(rhs_jac_transpose_mul, (p, y, v), "rhs_jac_transpose_mul.onnx")
to_onnx(rhs_sens_transpose_mul, (p, y, v), "rhs_sens_transpose_mul.onnx")

                        