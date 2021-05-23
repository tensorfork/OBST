import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_intermediate, get_variable, linear_from_features, linear_to_features, normal_var, \
    orthogonal_var
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import dropout as utils_dropout, einsum, greater_equal, sigmoid
from ..utils_mtf import anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def rezero(args: BlockArgs) -> mtf.Tensor:
    return args.tensor * get_variable(args.params, [], tf.constant_initializer(0))


def dropout(args: BlockArgs):
    keep = 1
    for extra in args.name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(args.tensor, args.params.train, keep)


def feed_forward_in(args: BlockArgs):
    def _from_feat():
        return linear_from_features(args, get_intermediate(args))

    out = dropout(args(activate(args(_from_feat()))))
    if 'glu' in args or 'glu_add' in args:
        out *= sigmoid(_from_feat())
    if 'glu_add' in args:
        out += activate(args(_from_feat()))
    return out


def feed_forward_out(args: BlockArgs):
    return linear_to_features(args, get_intermediate(args))


def feed_forward(args: BlockArgs) -> mtf.Tensor:
    return feed_forward_out(args(feed_forward_in(args)))


def spatial_mixing(args: BlockArgs) -> mtf.Tensor:
    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)

    if 'feed_forward' in args:
        args = args(activate(args(feed_forward_in(args))))
    if 'norm' in args:
        args = args(norm(args))

    mid = anonymize(args.tensor, dim)
    base = [args.params.head_dim] * ('group' in args)
    old = base + [tmp]
    new = base + [dim]

    inputs = [mid, orthogonal_var(args.params, old + new)]
    if is_masked(args):
        inputs.append(compare_range(args.params, dim, tmp, greater_equal))

    mid = einsum(inputs, args.tensor.shape)
    if 'multiply_gate' in args:
        if 'tanh' in args:
            mid = mtf.tanh(mid)
        elif 'sigmoid' in args:
            mid = mtf.sigmoid(mid)
        elif 'bias' in args:
            mid += normal_var(args.params, new, mean=1)
        mid *= args.tensor
    if 'feed_forward' not in args:
        return mid
    return feed_forward_out(args(mid))
