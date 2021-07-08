import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .dataclass import ModelParameter
from .utils_core import scoped

tf1 = tf.compat.v1
_NAME_INDEX = [0]

DIM = typing.Union[mtf.Dimension, str]
DIM_LIST = typing.List[mtf.Dimension]
SHAPE = typing.Union[mtf.Shape, DIM_LIST]
TENSORS = typing.List[mtf.Tensor]
OPT_SHAPE = typing.Optional[SHAPE]
OPT_DIMS = typing.Optional[DIM_LIST]
OPT_DIM = typing.Optional[mtf.Dimension]


def einsum(xs: TENSORS, output_shape: OPT_SHAPE = None, reduced_dims: OPT_DIMS = None) -> mtf.Tensor:
    return scoped("einsum", mtf.einsum, xs, output_shape, reduced_dims)


def one_hot(indices: mtf.Tensor, output_dim: mtf.Dimension, on_value: float = 1.0, off_value: float = 0.0,
            dtype: tf.dtypes = tf.float32) -> mtf.Tensor:
    return scoped("one_hot", mtf.one_hot, indices, output_dim, on_value, off_value, dtype)


def reduce_mean(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIM = None) -> mtf.Tensor:
    return scoped("reduce_mean", mtf.reduce_mean, tensor, None, output_shape, reduced_dim)


def reduce_sum(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIM = None) -> mtf.Tensor:
    return scoped("reduce_sum", mtf.reduce_sum, tensor, None, output_shape, reduced_dim)


def reduce_max(tensor: mtf.Tensor, output_shape: OPT_SHAPE = None, reduced_dim: OPT_DIM = None) -> mtf.Tensor:
    return scoped("reduce_max", mtf.reduce_max, tensor, None, output_shape, reduced_dim)


def reduce_logsumexp(tensor: mtf.Tensor, reduced_dim: OPT_DIM = None) -> mtf.Tensor:
    return scoped("reduce_logsumexp", mtf.reduce_logsumexp, tensor, reduced_dim)


def recompute_grad(fn: typing.Callable, explicit_inputs: typing.List[mtf.Tensor]) -> mtf.Tensor:
    return scoped("recompute_grad", mtf.recompute_grad, fn, explicit_inputs)


def stop_gradient(tensor: mtf.Tensor):
    return scoped("stop_gradient", mtf.stop_gradient, tensor)


def _softmax_cross_entropy_with_logits(params: ModelParameter, logits: mtf.Tensor, targets: mtf.Tensor):
    max_logit = reduce_max(stop_gradient(logits), reduced_dim=params.vocab_dim)
    log_z = add(log(reduce_sum(exp(add(logits, negative(max_logit))), reduced_dim=params.vocab_dim)), max_logit)
    loss = einsum([add(logits, negative(log_z)), one_hot(targets, params.vocab_dim, dtype=logits.dtype),
                   constant_scalar(params, -1 / targets.size)], output_shape=[])
    if not params.z_loss:
        return loss
    return add(loss, einsum([log_z, log_z, constant_scalar(params, params.z_loss / targets.size)], output_shape=[]))


def softmax_cross_entropy_with_logits(params: ModelParameter, logits: mtf.Tensor, targets: mtf.Tensor) -> mtf.Tensor:
    return scoped("softmax_cross_entropy_with_logits", _softmax_cross_entropy_with_logits, params, logits, targets)


def import_laid_out_tensor(params: ModelParameter, laid_out_tensor: object, shape: SHAPE,
                           name: typing.Optional[str] = None):
    return scoped("import_laid_out_tensor", mtf.import_laid_out_tensor, params.mesh, laid_out_tensor, shape, name)


def import_fully_replicated(params: ModelParameter, laid_out_tensor: tf.Tensor, shape: SHAPE,
                            name: typing.Optional[str] = None):
    return scoped("import_fully_replicated", mtf.import_fully_replicated, params.mesh, laid_out_tensor, shape, name)


def logical_not(tensor: mtf.Tensor):
    return scoped("logical_not", mtf.logical_not, tensor)


def logical_and(tensor: mtf.Tensor):
    return scoped("logical_and", mtf.logical_and, tensor)


def identity(tensor: mtf.Tensor):
    return scoped("identity", mtf.identity, tensor)


def while_loop(cond_fn: typing.Callable, body_fn: typing.Callable, inputs: TENSORS,
               num_loop_vars: typing.Optional[int] = None, has_accumulators: bool = False):
    return scoped("while_loop", mtf.while_loop, cond_fn, body_fn, inputs, num_loop_vars, has_accumulators)


def anonymize(tensor: mtf.Tensor):
    return scoped("anonymize", mtf.anonymize, tensor)


def random_uniform(params: ModelParameter, shape: SHAPE, dtype: typing.Optional[tf.DType] = None, maxval: float = 0,
                   minval: float = 0):
    return scoped("random_uniform", mtf.random_uniform, params.mesh, shape, dtype=dtype, maxval=maxval, minval=minval)


def relu(tensor: mtf.Tensor):
    return scoped("relu", mtf.relu, tensor)


def tanh(tensor: mtf.Tensor):
    return scoped("tanh", mtf.tanh, tensor)


def assign(var: mtf.Variable, new_val: mtf.Tensor):
    return scoped("assign", mtf.assign, var, new_val)


def assign_add(var: mtf.Variable, new_val: mtf.Tensor):
    return scoped("assign_add", mtf.assign_add, var, new_val)


def assign_sub(var: mtf.Variable, new_val: mtf.Tensor):
    return scoped("assign_sub", mtf.assign_sub, var, new_val)


def concat(tensors: typing.List[mtf.Tensor], concat_dim_name: str) -> mtf.Tensor:
    return scoped("concat", mtf.concat, tensors, concat_dim_name)


def pad(tensor: mtf.Tensor, padding: typing.Tuple[int, int], dim_name: str) -> mtf.Tensor:
    return scoped("concat", mtf.pad, tensor, padding, dim_name)


def constant(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None,
             dtype: typing.Union[None, mtf.VariableDType, tf.DType] = None) -> mtf.Tensor:
    return scoped("constant", mtf.constant, params.mesh, value, shape,
                  params.variable_dtype.activation_dtype if dtype is None else dtype)


def constant_float(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("constant_float", mtf.constant, params.mesh, value, shape, tf.float32)


def constant_int(params: ModelParameter, value: typing.Union[int, float], shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("constant_int", mtf.constant, params.mesh, value, shape, tf.int32)


def constant_scalar(params: ModelParameter, value: typing.Union[int, float], dtype: tf.DType = None) -> mtf.Tensor:
    dtype = params.variable_dtype.activation_dtype if dtype is None else dtype
    return scoped("constant_scalar", mtf.constant, params.mesh, value, [], dtype)


def optimizer_scalar(params: ModelParameter, value: typing.Union[int, float]) -> mtf.Tensor:
    return scoped("optimizer_scalar", mtf.constant, params.mesh, value, [], params.optimizer_calculation_dtype)


def greater_equal(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("greater_equal", mtf.greater_equal, x1, x2, output_shape)


def greater(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("greater", mtf.greater, x1, x2, output_shape)


def less(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("less", mtf.less, x1, x2, output_shape)


def less_equal(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("less_equal", mtf.less_equal, x1, x2, output_shape)


def equal(x1: mtf.Tensor, x2: mtf.Tensor, output_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("equal", mtf.equal, x1, x2, output_shape)


def mod(x1: mtf.Tensor, x2: typing.Union[mtf.Tensor, int]) -> mtf.Tensor:
    return scoped("mod", lambda x, y: x % y, x1, x2)


def sin(x: mtf.Tensor):
    return scoped("sin", mtf.sin, x)


def negative(tensor: mtf.Tensor):
    return scoped("negative", lambda x: -x, tensor)


def floordiv(x1: mtf.Tensor, x2: mtf.Tensor) -> mtf.Tensor:
    return scoped("floordiv", lambda x, y: x // y, x1, x2)


def mtf_range(mesh: mtf.Mesh, dim: DIM, dtype: tf.DType) -> mtf.Tensor:
    return scoped("range", mtf.range, mesh, dim, dtype)


def cast(tensor: mtf.Tensor, dtype: tf.DType) -> mtf.Tensor:
    return scoped("cast", mtf.cast, tensor, dtype)


def exp(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("exp", mtf.exp, tensor)


def reciprocal(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("reciprocal", mtf.reciprocal, tensor)


def log(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("log", mtf.log, tensor)


def reshape(tensor: mtf.Tensor, new_shape: SHAPE):
    return scoped("reshape", mtf.reshape, tensor, new_shape)


def argmax(tensor: mtf.Tensor, reduced_dim: mtf.Dimension):
    return scoped("argmax", mtf.argmax, tensor, reduced_dim)


def sigmoid(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("sigmoid", mtf.sigmoid, tensor)


def sqrt(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("sqrt", mtf.sqrt, tensor)


def sqrt_eps(tensor: mtf.Tensor, epsilon: float = 1e-6) -> mtf.Tensor:
    return scoped("sqrt", lambda x: sqrt(add(x, epsilon)), tensor)


def rsqrt(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("rsqrt", mtf.rsqrt, tensor)


def rsqrt_eps(tensor: mtf.Tensor, epsilon: float = 1e-6) -> mtf.Tensor:
    return scoped("rsqrt6", lambda x: rsqrt(add(x, epsilon)), tensor)


def softplus(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("softplus", mtf.softplus, tensor)


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


def mtf_slice(tensor: mtf.Tensor, begin: int, size: int, dim_name: str):
    return scoped("slice", mtf.slice, tensor, begin, size, dim_name)


def add(x1: mtf.Tensor, x2: mtf.Tensor):
    return scoped("add", lambda x, y: x + y, x1, x2)


def multiply(x1: mtf.Tensor, x2: mtf.Tensor):
    return scoped("multiply", lambda x, y: x * y, x1, x2, )


def divide(x1: mtf.Tensor, x2: float):
    return scoped("divide", lambda x, y: x / y, x1, x2)


def subtract(x1: mtf.Tensor, x2: mtf.Tensor):
    return scoped("subtract", lambda x, y: x - y, x1, x2)


def ones(mesh: mtf.Mesh, shape: SHAPE, dtype: tf.DType) -> mtf.Tensor:
    return scoped("ones", mtf.ones, mesh, shape, dtype)


def zeros(mesh: mtf.Mesh, shape: SHAPE, dtype: tf.DType) -> mtf.Tensor:
    return scoped("zeros", mtf.zeros, mesh, shape, dtype)


def pow(x1: mtf.Tensor, x2: mtf.Tensor) -> mtf.Tensor:
    return scoped("pow", mtf.pow, x1, x2)


def zeros_like(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("zeros_like", mtf.zeros_like, tensor)


def ones_like(tensor: mtf.Tensor) -> mtf.Tensor:
    return scoped("ones_like", mtf.ones_like, tensor)


def dropout(tensor: mtf.Tensor, is_training: bool, keep_prob: typing.Optional[float] = None,
            rate: typing.Optional[float] = None, noise_shape: OPT_SHAPE = None) -> mtf.Tensor:
    return scoped("dropout", mtf.dropout, tensor, is_training, keep_prob, rate, noise_shape)


def gradients(outputs: typing.List[mtf.Tensor], variables: typing.List[mtf.Tensor]):
    return scoped("gradients", mtf.gradients, outputs, variables)
