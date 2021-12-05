import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_var, linear, orthogonal_var
from .embedding import gather_embed
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import (dropout as utils_dropout, sigmoid, exp, reduce_max, reduce_sum, einsum, reciprocal, reshape,
                           multiply, reduce_mean)
from ..utils_mtf import linear_shapes, anonymize_shape, anonymize_dim, concat, get_dim

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


def reduced_half_linear(args: BlockArgs):
    return group_linear(args(reduce_mean(args.tensor, reduced_dim=args.params.head_dim)))


def product_key_memory(args: BlockArgs):
    two = mtf.Dimension("two", 2)
    out_shape = args.tensor.shape
    args = args(activated_linear_in(args))
    old, new = linear_shapes(args)
    features = [two, args.params.factorized_product_key_value_dim]
    assignment = linear(args, old, [args.params.head_dim] + features)
    assignment = norm(args(assignment), features)
    val0, idx0 = mtf.top_k(assignment, args.params.factorized_product_key_value_dim,
                           args.params.pkm_key_dim)
    val1, idx1 = mtf.top_k(-assignment, args.params.factorized_product_key_value_dim,
                           args.params.pkm_key_dim)
    val1 = -val1

    new_idx1 = mtf.slice(idx1, 0, 1, two.name) * args.params.factorized_product_key_value
    new_idx1 += anonymize_shape(mtf.slice(idx1, 1, 1, two.name), args.params.pkm_key_dim)
    new_val1 = mtf.slice(val1, 0, 1, two.name)
    new_val1 *= anonymize_shape(mtf.slice(val1, 1, 1, two.name), args.params.pkm_key_dim)

    new_idx0 = mtf.slice(idx0, 0, 1, two.name) * args.params.factorized_product_key_value
    new_idx0 += anonymize_shape(mtf.slice(idx0, 1, 1, two.name), args.params.pkm_key_dim)
    new_val0 = mtf.slice(val0, 0, 1, two.name)
    new_val0 *= anonymize_shape(mtf.slice(val0, 1, 1, two.name), args.params.pkm_key_dim)

    new_val0 = mtf.exp(new_val0 - mtf.stop_gradient(reduce_max(new_val0)))
    new_val1 = mtf.exp(new_val1 - mtf.stop_gradient(reduce_max(new_val1)))
    reduced = new_val0.shape - args.params.pkm_key_dim - anonymize_dim(args.params.pkm_key_dim)
    normalizer = mtf.reduce_sum(new_val0, output_shape=reduced) + mtf.reduce_sum(new_val1, output_shape=reduced)

    new_val0 /= normalizer
    new_val1 /= normalizer

    dim = mtf.Dimension("topk_squared", args.params.product_key_value_keys ** 2)
    new_val0 = mtf.reshape(new_val0, reduced + [dim])
    new_idx0 = mtf.reshape(new_idx0, reduced + [dim])
    new_val1 = mtf.reshape(new_val1, reduced + [dim])
    new_idx1 = mtf.reshape(new_idx1, reduced + [dim])
    val = concat([new_val0, new_val1], dim)
    idx = concat([new_idx0, new_idx1], dim)
    new_val, new_idx = mtf.top_k(val, get_dim(val, dim), args.params.pkm_key_dim)
    new_idx = mtf.gather(idx, new_idx, dim)

    out = gather_embed(args(new_idx), [args.params.product_key_value_dim] + args.params.feature_dims,
                       [args.params.head_dim])
    return mtf.einsum([out, new_val], out_shape)
