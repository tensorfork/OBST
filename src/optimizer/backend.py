import typing

import mesh_tensorflow as mtf
import tensorflow as tf2
from tensorflow.python.ops.init_ops import Initializer

from ..dataclass import ModelParameter
from ..mtf_wrapper import (import_fully_replicated)
from ..utils_mtf import SHAPE, get_variable

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def import_float(imported):
    return tf.constant(imported * 1.0, dtype=tf.float32, shape=[])


def get_var(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer = zeros):
    return get_variable(params, name, shape, initializer, False, params.optimizer_dtype)


def variable(params: ModelParameter, base: mtf.Variable, name: str, shape: SHAPE):
    return get_variable(params, f"{base.name}/{params.optimizer.replace(':', '_')}/{name}", shape, zeros, False,
                        params.optimizer_dtype)


def import_mtf(params: ModelParameter, imported: typing.Union[tf.Tensor, float], name: str):
    return import_fully_replicated(params, tf.cast(imported, params.optimizer_calculation_dtype), [], name)
