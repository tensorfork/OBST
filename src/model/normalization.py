import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import normal_var
from ..dataclass import BlockArgs
from ..mtf_wrapper import constant_scalar, einsum, reduce_mean, rsqrt
from ..utils_mtf import get_attention_dim

tf1 = tf.compat.v1


def norm(args: BlockArgs) -> mtf.Tensor:
    block_input = args.tensor
    normalized_shape = block_input.shape - [args.params.key_dim]

    if 'instance' not in args:
        normalized_shape = normalized_shape - [get_attention_dim(args).dim]
    if 'group' not in args:
        normalized_shape = normalized_shape - [args.params.head_dim]

    if 'mean' in args:
        block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = []
    if 'std' in args:
        scale.append(rsqrt(1e-6 + einsum([block_input, block_input,
                                          constant_scalar(args.params, normalized_shape.size / block_input.size)],
                                         output_shape=normalized_shape)))
    if 'scale' in args:
        scale.append(normal_var(args.params, args.params.feature_dims, mean=1))
    if scale:
        block_input = mtf.einsum([block_input] + scale, output_shape=block_input.shape)
    if 'shift' in args:
        block_input += normal_var(args.params, args.params.feature_dims, mean=0)
    return block_input