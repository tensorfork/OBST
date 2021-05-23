import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .dataclass import ModelParameter
from .utils_core import random_name

tf1 = tf.compat.v1
_NAME_INDEX = [0]

DIM = typing.Union[mtf.Dimension, str]
DIM_LIST = typing.List[mtf.Dimension]
SHAPE = typing.Union[mtf.Shape, DIM_LIST]
TENSORS = typing.List[mtf.Tensor]
OPT_SHAPE = typing.Optional[SHAPE]
OPT_DIMS = typing.Optional[DIM_LIST]


def scoped(name: str, fn: typing.Callable, *args, **kwargs):
    with tf1.variable_scope(random_name(name)):
        return fn(*args, **kwargs)


def einsum(xs: TENSORS, output_shape: OPT_SHAPE = None, reduced_dims: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("einsum", mtf.einsum, xs, output_shape, reduced_dims)


def one_hot(indices: mtf.Tensor, output_dim: mtf.Dimension, on_value: float = 1.0, off_value: float = 0.0,
            dtype: tf.dtypes = tf.float32) -> mtf.Tensor:
    return scoped("one_hot", mtf.one_hot, indices, output_dim, on_value, off_value, dtype)


def reduce_mean(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("reduce_mean", mtf.reduce_mean, tensor, None, output_shape, reduced_dim)


def reduce_sum(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("reduce_sum", mtf.reduce_sum, tensor, None, output_shape, reduced_dim)


def reduce_max(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("reduce_max", mtf.reduce_max, tensor, None, output_shape, reduced_dim)


def reduce_logsumexp(tensor: mtf.Tensor, reduced_dim: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("reduce_logsumexp", mtf.reduce_logsumexp, tensor, reduced_dim)


def constant(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("constant", mtf.constant, params.mesh, value, shape, params.variable_dtype.activation_dtype)


def constant_float(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("constant_float", mtf.constant, params.mesh, value, shape, tf.float32)


def constant_int(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("constant_int", mtf.constant, params.mesh, value, shape, tf.int32)


def constant_scalar(params: ModelParameter, value: typing.Union[int, float], dtype: tf.TypeSpec = None) -> mtf.Tensor:
    dtype = params.variable_dtype.activation_dtype if dtype is None else dtype
    return scoped("constant_scalar", mtf.constant, params.mesh, value, [], dtype)


def greater_equal(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("greater_equal", mtf.greater_equal, x1, x2, output_shape)


def greater(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("greater", mtf.greater, x1, x2, output_shape)


def less(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("less", mtf.less, x1, x2, output_shape)


def equal(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("equal", mtf.equal, x1, x2, output_shape)


def mod(x1: mtf.Tensor, x2: typing.Union[mtf.Tensor, int], output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("mod", mtf.mod, x1, x2, output_shape)


def sin(x: mtf.Tensor):
    return scoped("sin", mtf.sin, x)


def floordiv(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("floordiv", mtf.floordiv, x1, x2, output_shape)


def mtf_range(mesh: mtf.Mesh, dim: DIM, dtype: tf.dtypes) -> mtf.Tensor:
    return scoped("range", mtf.range, mesh, dim, dtype)


def cast(tensor: mtf.Tensor, dtype: tf.dtypes) -> mtf.Tensor:
    return scoped("cast", mtf.cast, tensor, dtype)


def exp(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("exp", mtf.exp, tensor)


def reciprocal(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("reciprocal", mtf.reciprocal, tensor)


def log(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("log", mtf.log, tensor)


def sigmoid(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("sigmoid", mtf.sigmoid, tensor)


def sqrt(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("sqrt", mtf.sqrt, tensor)


def rsqrt(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("rsqrt", mtf.rsqrt, tensor)


def square(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("square", mtf.square, tensor)


def sign(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("sign", mtf.sign, tensor)


def shift(tensor: mtf.Tensor, offset: int, dim: DIM, wrap: bool) -> mtf.Tensor:
    return scoped("shift", mtf.shift, tensor, offset, dim, wrap)


def maximum(x1: mtf.Tensor, x2: typing.Union[mtf.Tensor, int, float], output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("maximum", mtf.maximum, x1, x2, output_shape)


def minimum(x1: mtf.Tensor, x2: typing.Union[mtf.Tensor, int, float], output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("minimum", mtf.minimum, x1, x2, output_shape)


def add_n(*xs: typing.Union[typing.List[TENSORS], TENSORS]) -> mtf.Tensor:
    if len(xs) == 1 and not isinstance(xs[0], mtf.Tensor):
        xs = xs[0]
    return scoped("add_n", mtf.add_n, xs)


def ones(mesh: mtf.Mesh, shape: SHAPE, dtype: tf.dtypes) -> mtf.Tensor:
    return scoped("ones", mtf.ones, mesh, shape, dtype)


def zeros(mesh: mtf.Mesh, shape: SHAPE, dtype: tf.dtypes) -> mtf.Tensor:
    return scoped("zeros", mtf.zeros, mesh, shape, dtype)


def zeros_like(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("zeros_like", mtf.zeros_like, tensor)


def dropout(tensor: mtf.Tensor, is_training: bool, keep_prob: typing.Optional[float] = None,
            rate: typing.Optional[float] = None, noise_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("dropout", mtf.dropout, tensor, is_training, keep_prob, rate, noise_shape)
