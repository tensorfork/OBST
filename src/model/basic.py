import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate_util
from .backend import (get_attention_dim, get_intermediate, get_variable, linear_from_features, linear_to_features,
                      orthogonal_var)
from .normalization import norm
from ..dataclass import ModelParameter
from ..mtf_wrapper import dropout as utils_dropout, einsum, sigmoid
from ..utils_mtf import anonymize, anonymize_dim, deduplicate

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def rezero(params, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return block_input * get_variable(params, [], tf.constant_initializer(0))


def dropout(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    keep = 1
    for extra in name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(block_input, keep)


def feed_forward_in(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List):
    def _from_feat():
        return linear_from_features(params, block_input, get_intermediate(params, name_extras))

    out = dropout(params, activate_util(name_extras, _from_feat()), name_extras)
    if 'glu' in name_extras or 'glu_add' in name_extras:
        out *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        out += activate_util(name_extras, _from_feat())
    return out


def feed_forward_out(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List):
    return linear_to_features(params, block_input, get_intermediate(params, name_extras))


def feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return feed_forward_out(params, feed_forward_in(params, block_input, name_extras), name_extras)


def spatial_mixing(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    dim = get_attention_dim(params, block_input)
    tmp = anonymize_dim(dim)

    if 'feed_forward' in name_extras:
        mid = feed_forward_in(params, block_input, name_extras)
    else:
        mid = block_input

    if 'norm' in name_extras:
        mid = norm(params, block_input, name_extras + ['group'])
    mid = anonymize(mid, dim)
    old = [params.head_dim, tmp]
    new = [params.head_dim, dim]
    mid = einsum([mid, block_input, orthogonal_var(params, old + new)],
                 deduplicate((block_input.shape - old).dims + new))
    if 'feed_forward' not in name_extras:
        return mid
    return feed_forward_out(params, block_input, name_extras)
