import math
import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_var, linear, orthogonal_var
from .embedding import gather_embed
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import (dropout as utils_dropout, sigmoid, exp, reduce_max, reduce_sum, einsum, reciprocal, reshape,
                           multiply, reduce_mean, stop_gradient)
from ..utils_mtf import linear_shapes, anonymize_shape, unbind, replace_dim

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def rezero(args: BlockArgs) -> mtf.Tensor:
    return args.tensor * get_var(args, [], tf.constant_initializer(0))


def dropout(args: BlockArgs):
    keep = 1
    for extra in args.name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(args.tensor, args.params.train, keep)


def wrapped_linear(args: BlockArgs) -> mtf.Tensor:
    return linear(args, *linear_shapes(args))


def mixture_of_experts(args: BlockArgs) -> mtf.Tensor:
    old, new = linear_shapes(args)
    gate = linear(args, old, [args.params.expert_dim])
    gate -= mtf.stop_gradient(reduce_max(gate, reduced_dim=args.params.expert_dim))
    gate = exp(gate)
    return einsum([reciprocal(reduce_sum(gate, reduced_dim=args.params.expert_dim)), args.tensor, gate,
                   orthogonal_var(args, old + new + [args.params.expert_dim])],
                  output_shape=args.tensor.shape - old + new)


def activated_linear(args: BlockArgs, prefix: str) -> mtf.Tensor:
    args = args([a[len(prefix):] for a in args if a.startswith(prefix)])
    feed_forward_fn = mixture_of_experts if 'mixture_of_experts' in args else wrapped_linear
    out = dropout(args(activate(args(feed_forward_fn(args)))))
    if 'glu' in args or 'glu_add' in args:
        out = multiply(out, sigmoid(feed_forward_fn(args)))
    if 'glu_add' in args:
        out += activate(args(feed_forward_fn(args)))
    if 'norm' in args:
        out = norm(args(out))
    return out


def activated_linear_in(args: BlockArgs):
    return activated_linear(args, 'in:')


def activated_linear_out(args: BlockArgs):
    return activated_linear(args, 'out:')


def feed_forward(args: BlockArgs) -> mtf.Tensor:
    return activated_linear_out(args(activated_linear_in(args)))


def group_linear(args: BlockArgs):
    return reshape(linear(args('group'), args.params.feature_dims,
                          anonymize_shape(args.params.feature_dims, args.params.key_dim)),
                   args.tensor.shape - args.params.feature_dims + args.params.feature_dims)


def sum_heads(args: BlockArgs):
    return reduce_sum(args.tensor, reduced_dim=args.params.head_dim)


def transpose_sequence_features(args: BlockArgs):
    shape = list(args.tensor.shape)
    intermediate = mtf.Dimension("intermediate", 1)
    shape = replace_dim(shape, intermediate, args.params.sequence_dim)
    shape = replace_dim(shape, args.params.sequence_dim, args.params.key_dim)
    shape = replace_dim(shape, args.params.key_dim, intermediate)
    return reshape(args.tensor, shape)


def reduced_half_linear(args: BlockArgs):
    return group_linear(args(reduce_mean(args.tensor, reduced_dim=args.params.head_dim)))


def product_key_memory(args: BlockArgs):
    args = args(activated_linear_in(args))
    old, new = linear_shapes(args)
    features = [args.params.pkm_dim, args.params.key_dim]
    assignment = linear(args, old, [args.params.head_dim] + features)
    assignment = norm(args(assignment), features)
    assignment -= stop_gradient(reduce_max(assignment,
                                           output_shape=assignment.shape - [args.params.pkm_dim, args.params.key_dim]))
    assignment = exp(assignment)
    normalizer = reduce_sum(assignment, output_shape=assignment.shape - [args.params.key_dim])
    normalizer = einsum(unbind(normalizer, args.params.pkm_dim), output_shape=normalizer.shape - args.params.pkm_dim)

    val, idx = mtf.top_1(assignment, args.params.key_dim)
    idx = mtf.einsum([mtf.cast(exp(math.log(args.params.features_per_head) *
                                   mtf.range(normalizer.mesh, args.params.pkm_dim, dtype=normalizer.dtype)),
                               tf.int32), idx], output_shape=idx.shape - args.params.pkm_dim)
    val = einsum(unbind(val, args.params.pkm_dim), output_shape=val.shape - args.params.pkm_dim) / normalizer
    out = gather_embed(args(idx), [args.params.product_key_value_dim] + args.params.feature_dims,
                       [args.params.head_dim])
    return out * val
