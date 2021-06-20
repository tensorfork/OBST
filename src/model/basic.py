import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_var, linear, orthogonal_var
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import dropout as utils_dropout, sigmoid, exp, reduce_max, reduce_sum, einsum, reciprocal
from ..utils_mtf import linear_shapes

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
    gate = exp(gate - reduce_max(gate, reduced_dim=args.params.expert_dim))
    return einsum([reciprocal(reduce_sum(gate, reduced_dim=args.params.expert_dim)), args.tensor, gate,
                   orthogonal_var(args, old + new + [args.params.expert_dim])],
                  output_shape=args.tensor.shape - old + new)


def activated_linear(args: BlockArgs, prefix: str) -> mtf.Tensor:
    args = args([a[len(prefix):] for a in args if a.startswith(prefix)])
    feed_forward_fn = mixture_of_experts if 'mixture_of_experts' in args else wrapped_linear
    out = dropout(args(activate(args(feed_forward_fn(args)))))
    if 'glu' in args or 'glu_add' in args:
        out *= sigmoid(feed_forward_fn(args))
    if 'glu_add' in args:
        out += activate(args(feed_forward_fn(args)))
    if 'norm' in args:
        out = norm(args(out))
    return out


def activated_linear_in(args: BlockArgs):
    return activated_linear(args, 'in_')


def activated_linear_out(args: BlockArgs):
    return activated_linear(args, 'out_')


def feed_forward(args: BlockArgs) -> mtf.Tensor:
    return activated_linear_out(args(activated_linear_in(args)))
