import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from src.dataclass import ModelParameter
from src.utils_mtf import (SHAPE, activate as utils_activate, anonymize_dim, constant_scalar, einsum, one_hot,
                           reduce_mean, rsqrt,
                           scoped, sigmoid)
from .backend import get_attention_dim, get_variable, linear_from_features, linear_to_features, normal_var

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def norm(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    normalized_shape = block_input.shape - [params.key_dim]
    if 'instance' not in name_extras:
        normalized_shape = normalized_shape - [get_attention_dim(params, block_input).dim]
    if 'group' not in name_extras:
        normalized_shape = normalized_shape - [params.head_dim]
    if 'mean' in name_extras:
        block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = []
    if 'std' in name_extras:
        scale.append(rsqrt(1e-6 + einsum([block_input, block_input,
                                          constant_scalar(params, normalized_shape.size / block_input.size)],
                                         output_shape=normalized_shape)))
    if 'scale' in name_extras:
        scale.append(normal_var(params, params.feature_dims, mean=1))
    if scale:
        block_input = mtf.einsum([block_input] + scale, output_shape=block_input.shape)
    if 'shift' in name_extras:
        block_input += normal_var(params, params.feature_dims, mean=0)
    return block_input


def rezero(params, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return block_input * get_variable(params, [], tf.constant_initializer(0))


def embed(params: ModelParameter, shape: SHAPE) -> mtf.Tensor:
    return scoped("embed", normal_var, params, shape, params.embedding_stddev)


def all_mean(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple):
    return einsum([block_input,
                   one_hot(constant_scalar(params, get_attention_dim(params, block_input).index / block_input.size),
                           params.head_dim)],
                  reduced_dims=[params.head_dim])


def feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    if 'group' in name_extras:
        intermediate = [params.head_dim,
                        anonymize_dim(params.key_dim, params.key_dim.size * params.group_linear_factor)]
    else:
        intermediate = params.intermediate

    def _from_feat():
        return linear_from_features(params, block_input, intermediate)

    mid = utils_activate(name_extras, _from_feat())
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += utils_activate(name_extras, _from_feat())
    return linear_to_features(params, mid, intermediate)


def activate(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    return utils_activate(name_extras, block_input)
