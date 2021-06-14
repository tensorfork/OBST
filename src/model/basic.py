import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_intermediate, get_variable, linear_from_features, linear_to_features, linear
from .normalization import norm
from ..dataclass import BlockArgs
from ..mtf_wrapper import dropout as utils_dropout, sigmoid
from ..utils_mtf import anonymize_dim, get_attention_dim, replace_dim

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def rezero(args: BlockArgs) -> mtf.Tensor:
    return args.tensor * get_variable(args, [], tf.constant_initializer(0))


def dropout(args: BlockArgs):
    keep = 1
    for extra in args.name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(args.tensor, args.params.train, keep)


def from_feat(args: BlockArgs):
    return linear_from_features(args, get_intermediate(args))


def activate_norm(args: BlockArgs, feed_forward_fn: typing.Callable) -> mtf.Tensor:
    out = dropout(args(activate(args(feed_forward_fn(args)))))
    if 'glu' in args or 'glu_add' in args:
        out *= sigmoid(feed_forward_fn(args))
    if 'glu_add' in args:
        out += activate(args(feed_forward_fn(args)))
    if 'norm' in args:
        out = norm(args(out))
    return out


def expert_from_feat(args: BlockArgs):
    params = args.params
    return linear(args, [params.expert_dim] + params.feature_dims, [params.experts] + get_intermediate(args))


def expert_to_feat(args: BlockArgs):
    params = args.params
    return linear(args, [params.experts] + get_intermediate(args), [params.expert_dim] + params.feature_dims)


def feed_forward_in(args: BlockArgs) -> mtf.Tensor:
    return activate_norm(args, from_feat)


def feed_forward_out(args: BlockArgs):
    return linear_to_features(args, get_intermediate(args))


def feed_forward(args: BlockArgs) -> mtf.Tensor:
    return feed_forward_out(args(feed_forward_in(args)))


def mixture_of_experts(args: BlockArgs) -> mtf.Tensor:
    experts = args.params.experts
    dim = get_attention_dim(args).dim
    if dim.size % experts:
        raise ValueError(f"Make sure that {dim} is divisible by number of experts ({experts})")
    tensor = replace_dim(args.tensor, [anonymize_dim(dim, dim.size // experts), args.params.expert_dim], dim)
    tensor = expert_to_feat(args(activate_norm(args(tensor), expert_from_feat)))
    return mtf.reshape(tensor, args.tensor.shape)
