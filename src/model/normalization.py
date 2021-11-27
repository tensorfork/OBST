import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import normal_var, SHAPE
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, reduce_mean, rsqrt_eps, square
from ..utils_mtf import linear_shapes

tf1 = tf.compat.v1


def norm(args: BlockArgs, feature_shape: typing.Optional[SHAPE] = None) -> mtf.Tensor:
    block_input = args.tensor
    feature_shape = mtf.Shape(linear_shapes(args).old if feature_shape is None else feature_shape)
    normalized_shape = block_input.shape - (feature_shape - [args.params.head_dim] * ('group' in args))
    if 'proxy' in args:
        base = normal_var(args, feature_shape, mean=0)
        sub = reduce_mean(base, output_shape=[])
        base -= sub
        block_input -= sub
        div = rsqrt_eps(reduce_mean(square(base), output_shape=[]), 1e-5)
    else:
        block_input -= reduce_mean(block_input, output_shape=normalized_shape)
        div = rsqrt_eps(reduce_mean(square(block_input), output_shape=normalized_shape), 1e-5)
    scale = [div, block_input]
    if 'scale' in args:
        scale.append(normal_var(args, feature_shape, mean=1))
    block_input = einsum(scale, output_shape=block_input.shape)
    if 'shift' in args:
        block_input += normal_var(args, feature_shape, mean=0)
    return block_input
