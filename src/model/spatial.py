import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .basic import activated_linear_in, activated_linear_out
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, greater_equal, multiply, add, less, exp, reduce_max, reciprocal, reduce_sum, negative
from ..utils_mtf import (anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked, linear_shapes)

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def _masked_map(args: BlockArgs):
    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)
    bias = embed(args, [args.params.head_dim, dim, tmp])
    return bias, compare_range(args.params, dim, tmp, greater_equal) if is_masked(args) else 1


def attention(args: BlockArgs):
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
            key = add(key, embed(args, [dim] + args.params.feature_dims))
        qry = activated_linear_out(base)
        qry = multiply(qry, dim.size ** -0.5)
        logit_shape = shape - (mtf.Shape(linear_shapes(args).old) - [args.params.head_dim]) + anonymize_dim(dim)
        logit = einsum([qry, anonymize(key, dim)], output_shape=logit_shape)
        if "shared_key_value" in args:
            val = key
    if 'biased_softmax' in args:
        logit = add(logit, multiply(*_masked_map(args)))
    if logit != 0:
        logit = add(logit, multiply(multiply(compare_range(args.params, dim, tmp, less), 1e38), -2))
        logit = add(logit, negative(reduce_max(logit, reduced_dim=tmp)))
        logit = exp(logit)
        logit = multiply(logit, reciprocal(reduce_sum(logit, reduced_dim=tmp)))
    if 'biased_attention_map' in args and logit != 0 and "scale_attention_map" not in args:
        logit = add(logit, multiply(*_masked_map(args)))
    logit = [logit] * (logit != 0)
    if 'scale_attention_map' in args or ("biased_attention_map" in args and not logit):
        logit.extend(_masked_map(args))
    if val == 0:
        val = anonymize(args.tensor if "input_as_value" in args else activated_linear_out(base), dim)
    if not logit:
        raise UserWarning(f"WARNING: There is no spatial mixing happening with the following attention parameters: "
                          f"{args.name_extras}.")
    return einsum(logit + [val], shape)
