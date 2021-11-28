import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from scipy.special import erfinv

from .backend import normal_var, SHAPE
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, reduce_mean, rsqrt_eps, square
from ..utils_mtf import linear_shapes

tf1 = tf.compat.v1


def uniformly_sampled_gaussian(num_rand):
    rand = 2 * (np.arange(num_rand) + 0.5) / float(num_rand) - 1
    return np.sqrt(2) * erfinv(rand)


def norm(args: BlockArgs, feature_shape: typing.Optional[SHAPE] = None) -> mtf.Tensor:
    block_input = args.tensor
    feature_shape = mtf.Shape(linear_shapes(args).old if feature_shape is None else feature_shape)
    normalized_shape = block_input.shape - (feature_shape - [args.params.head_dim] * ('group' in args))

    scale = normal_var(args, feature_shape, mean=1) if 'scale' in args else None
    shift = normal_var(args, feature_shape, mean=0) if 'shift' in args else None

    if 'proxy' in args:
        proxy_z = mtf.constant(block_input.mesh, uniformly_sampled_gaussian(args.params.train_batch_size),
                               [args.params.batch_dim], dtype=block_input.dtype)
        proxy_z *= scale
        proxy_z += shift
        sub = reduce_mean(proxy_z, output_shape=[])
        proxy_z -= sub
        if shift:
            block_input -= shift
        if scale:
            block_input /= scale
        block_input -= proxy_z
        block_input /= rsqrt_eps(reduce_mean(square(proxy_z), output_shape=[]), 1e-5)
    else:
        block_input -= reduce_mean(block_input, output_shape=normalized_shape)
        div = rsqrt_eps(reduce_mean(square(block_input), output_shape=normalized_shape), 1e-5)
        scale = ([scale] * bool(scale)) + [div, block_input]
        block_input = einsum(scale, output_shape=block_input.shape)
        if shift:
            block_input += shift
    return block_input
