import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .basic import activated_linear_in, activated_linear_out
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, greater_equal,greater, less,  multiply, scoped, add, reduce_sum, multiply, exp, reduce_max, reciprocal
from ..utils_core import random_name
from ..utils_mtf import (anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked, linear_shapes, compare_range)

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))


def attention(args: BlockArgs):
    args.params.attention_idx += 1

    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)
    shape = args.tensor.shape
    logit_shape = args.tensor.shape - (mtf.Shape(linear_shapes(args).old) - [args.params.head_dim]) + anonymize_dim(dim)

    base = args(activated_linear_in(args))

    qry = activated_linear_out(base)
    key = activated_linear_out(base)
    val = activated_linear_out(base)

    # key = add(key, embed(args, [dim] + args.params.feature_dims))
    key = anonymize(key, dim)
    qry = multiply(qry, dim.size ** -0.5)
    val = anonymize(val, dim)

    logit = einsum([qry, key], output_shape=logit_shape)

    logit = add(logit, multiply(multiply(compare_range(args.params, dim, tmp, less), 1e38), -2))
    logit = add(logit, -reduce_max(logit, reduced_dim=tmp))
    logit = exp(logit)
    logit = multiply(logit, reciprocal(reduce_sum(logit, reduced_dim=tmp)))

    return einsum([logit, val], shape)
