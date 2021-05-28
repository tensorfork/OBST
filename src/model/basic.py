import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_intermediate, get_variable, linear_from_features, linear_to_features
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import dropout as utils_dropout, sigmoid

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


def full_feed_forward_in(args: BlockArgs) -> mtf.Tensor:
    out = activate(args(feed_forward_in(args)))
    return dropout(args(norm(args(out)) if 'norm' in args else out))
