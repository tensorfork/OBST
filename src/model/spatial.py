import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .basic import activated_linear_in, activated_linear_out
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, greater_equal, multiply, less, exp, reduce_max, reduce_sum
from ..utils_mtf import (anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked, linear_shapes,
                         random_name)

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def _masked_map(args: BlockArgs) -> typing.Tuple[mtf.Tensor, typing.Union[mtf.Tensor, int]]:
    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)
    bias = embed(args, [args.params.head_dim, dim, tmp])
    return bias, compare_range(args.params, dim, tmp, greater_equal) if is_masked(args) else 1


def cumsum(args: BlockArgs) -> mtf.Tensor:
    dim = args.tensor.shape.dims.index(get_attention_dim(args).dim)
    return mtf.cwise(lambda x: tf.cumsum(x, dim), [args.tensor], name=random_name("cumsum"),
                     grad_function=lambda _, dy: tf.reverse(tf.cumsum(tf.reverse(dy, dim), dim), dim))


def cummean(args: BlockArgs) -> mtf.Tensor:
    return cumsum(args) / (1 + mtf.range(args.tensor.mesh, get_attention_dim(args).dim, dtype=args.tensor.dtype,
                                         name=random_name("cummean")))


def attention(args: BlockArgs) -> mtf.Tensor:
    args.params.attention_idx += 1
    if "dot_product" in args or "input_as_value" not in args:
        base = args(activated_linear_in(args))

    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)
    shape = args.tensor.shape

    logit = 0
    val = 0
    key = 0
    if 'dot_product' in args:
        if 'embedded' in args or 'context' in args:
            key = activated_linear_out(base)
        if 'embedded' in args or 'positional' in args:
            key += embed(args, [dim] + args.params.feature_dims)
        qry = activated_linear_out(base)
        qry *= dim.size ** -0.5
        logit_shape = shape - (mtf.Shape(linear_shapes(args).old) - [args.params.head_dim]) + tmp
        logit = einsum([qry, anonymize(key, dim)], output_shape=logit_shape)
        if "shared_key_value" in args:
            val = key
    if 'biased_softmax' in args:
        logit += multiply(*_masked_map(args))
    if logit != 0:
        logit += (compare_range(args.params, dim, tmp, less) * 1e38) * -2
        logit -= mtf.stop_gradient(reduce_max(logit, reduced_dim=tmp))
        logit = exp(logit)
        logit /= reduce_sum(logit, reduced_dim=tmp)
    if 'biased_attention_map' in args:
        logit += multiply(*_masked_map(args))
    if 'scale_attention_map' in args:
        logit *= multiply(*_masked_map(args))
    if val == 0:
        val = anonymize(args.tensor if "input_as_value" in args else activated_linear_out(base), dim)
    if not logit:
        raise UserWarning(f"WARNING: There is no spatial mixing happening with the following attention parameters: "
                          f"{args.name_extras}.")
    return einsum([logit, val], shape)
