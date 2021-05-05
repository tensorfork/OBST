import random
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops, gen_linalg_ops, math_ops, random_ops
from tensorflow.python.ops.init_ops import Initializer

from ..dataclass import ModelParameter
from ..utils_core import default
from ..utils_mtf import OPT_DIMS, SHAPE, deduplicate, feature_dims_used
from ..utils_core import random_name
from ..mtf_wrapper import einsum, scoped

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


class OrthogonalInit(Initializer):
    def __init__(self, params: ModelParameter, shape: SHAPE, fan_in_dims: OPT_DIMS = None):
        if fan_in_dims is None:
            fan_in_dims = []
        self.params = params
        self.sizes = [d.size for d in shape]
        self.seed = random.randint(0, 2 ** 32)
        sizes = [d.size for d in mtf.Shape(shape) - fan_in_dims]
        features_used = feature_dims_used(params, shape)
        if fan_in_dims is None:
            if features_used:
                if shape.index(params.key_dim) == len(sizes) - 1:
                    fan_in = np.prod(sizes[:-2])
                else:
                    fan_in = np.prod([d.size for d in params.feature_dims])
            elif len(sizes) == 2:
                fan_in = sizes[0]
            else:
                raise ValueError(f"Shape: {shape}\nParams: {params}\nFeaturesUsed: {features_used}")
        else:
            fan_in = int(np.prod([d.size for d in fan_in_dims]))
        fan_out = np.prod(sizes) // fan_in
        self.transpose = transpose = fan_out > fan_in
        self.shape = (fan_out, fan_in) if transpose else (fan_in, fan_out)

    def __call__(self, shape, dtype=None, partition_info=None):
        q, r = gen_linalg_ops.qr(random_ops.random_normal(self.shape, dtype=tf.float32, seed=self.seed))
        q *= math_ops.sign(array_ops.diag_part(r))
        if self.transpose:
            q = array_ops.matrix_transpose(q)
        return tf.cast(array_ops.reshape(q, self.sizes) / self.params.n_blocks ** 0.5, dtype)


def get_variable(params: ModelParameter, shape: SHAPE, initializer: typing.Callable) -> mtf.Tensor:
    return scoped(random_name("get_variable"), mtf.get_variable, params.mesh, random_name("get_variable"),
                  deduplicate(shape), dtype=params.variable_dtype, initializer=initializer)


def orthogonal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                   fan_in_dims: OPT_DIMS = None) -> mtf.Tensor:
    shape = deduplicate(shape)
    return scoped("orthogonal_var", get_variable, params, shape, OrthogonalInit(params, shape, fan_in_dims))


def normal_var(params: ModelParameter, shape: SHAPE, stddev: float = 0.02, mean: float = 0.) -> mtf.Tensor:
    shape = deduplicate(shape)
    return scoped("normal_var", get_variable, params, shape, tf.random_normal_initializer(stddev=stddev, mean=mean))


def get_attention_dim(params: ModelParameter, block_input: typing.Union[mtf.Tensor, mtf.Shape]) -> ATTENTION_DIM:
    if isinstance(block_input, mtf.Tensor):
        block_input = block_input.shape
    attention_dims = (block_input - params.feature_dims - params.intermediate)[1:]  # Ex: Shape[Sequence, Width, Height]
    idx = params.attention_idx % len(attention_dims)
    dim = attention_dims[idx]
    return ATTENTION_DIM(idx, dim)


def linear(params: ModelParameter, block_input: mtf.Tensor, old: typing.List[mtf.Dimension],
           new: typing.List[mtf.Dimension]) -> mtf.Tensor:
    return einsum([block_input, orthogonal_var(params, old + new)],
                  deduplicate((block_input.shape - old).dims + new))


def linear_to_features(params: ModelParameter, block_input: mtf.Tensor,
                       old: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return linear(params, block_input, default(old, params.feature_dims), params.feature_dims)


def linear_from_features(params: ModelParameter, block_input: mtf.Tensor,
                         new: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return linear(params, block_input, params.feature_dims, default(new, params.intermediate))


def communicating_linear(params: ModelParameter, block_input: mtf.Tensor):
    return linear_to_features(params, block_input, params.intermediate)
