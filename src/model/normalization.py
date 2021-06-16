import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import normal_var
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, reduce_mean, rsqrt
from ..utils_mtf import shape_crossection, get_intermediate

tf1 = tf.compat.v1


def norm(args: BlockArgs) -> mtf.Tensor:
    block_input = args.tensor
    feature_shape = shape_crossection(block_input.shape, args.params.feature_dims + get_intermediate(args))
    normalized_shape = block_input.shape - (feature_shape - [args.params.head_dim] * ('group' in args))

    block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = [rsqrt(mtf.reduce_mean(mtf.square(block_input), output_shape=normalized_shape) + 1e-6), block_input]
    if 'scale' in args:
        scale.append(normal_var(args, feature_shape, mean=1))
    block_input = einsum(scale, output_shape=block_input.shape)
    if 'shift' in args:
        block_input += normal_var(args, feature_shape, mean=0)
    return block_input
