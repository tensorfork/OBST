import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import normal_var
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, reduce_mean, rsqrt_eps, square, add, negative
from ..utils_mtf import linear_shapes

tf1 = tf.compat.v1


def norm(args: BlockArgs) -> mtf.Tensor:
    block_input = args.tensor
    feature_shape = mtf.Shape(linear_shapes(args).old)
    normalized_shape = block_input.shape - (feature_shape - [args.params.head_dim] * ('group' in args))

    block_input = add(block_input, negative(reduce_mean(block_input, output_shape=normalized_shape)))
    scale = [rsqrt_eps(reduce_mean(square(block_input), output_shape=normalized_shape)), block_input]
    if 'scale' in args:
        scale.append(normal_var(args, feature_shape, mean=1))
    block_input = einsum(scale, output_shape=block_input.shape)
    if 'shift' in args:
        block_input = add(block_input, normal_var(args, feature_shape, mean=0))
    return block_input
