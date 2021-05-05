import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .activation import activate_util
from .backend import get_attention_dim, get_variable, linear_from_features, linear_to_features, normal_var
from ..dataclass import ModelParameter
from ..mtf_wrapper import (add_n, constant_scalar, einsum, exp, mod, mtf_range, one_hot, reduce_mean, rsqrt, scoped,
                           sigmoid, sin)
from ..utils_mtf import DIM_LIST, SHAPE, anonymize_dim, shape_size

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


def _multi_dim_range(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    return add_n([mtf_range(params.mesh, dim, params.variable_dtype.activation_dtype) * size
                  for dim, size in zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]]))])


_EMBEDDINGS = {}


def embed(params: ModelParameter, shape: SHAPE) -> mtf.Tensor:
    position_dims: SHAPE = shape - params.feature_dims - params.intermediate
    feature_dims = list(set(shape.dims).union(set(params.feature_dims + params.intermediate)))
    position_count = shape_size(position_dims)
    feature_count = shape_size(feature_dims)

    op = mtf.add if 'additive' in params.position_embedding else mtf.multiply
    absolute = 'absolute' in params.position_embedding
    split = 'split' in params.position_embedding
    relative = 'relative' in params.position_embedding
    learned = 'learned' in params.position_embedding
    cosine = 'cosine' in params.position_embedding
    axial = 'axial' in params.position_embedding

    out = None

    if params.shared_position_embedding and shape in _EMBEDDINGS:
        return _EMBEDDINGS[shape]

    if absolute and split:
        out = op(normal_var(params, position_dims, params.embedding_stddev),
                 normal_var(params, feature_dims, params.embedding_stddev))
    if absolute:
        out = scoped("absolute_posembed", normal_var, params, shape, params.embedding_stddev)

    if relative and learned and split:
        positions = _multi_dim_range(params, position_dims)
        out = op(positions, normal_var(params, feature_dims, 1 / position_count))
    elif relative:
        positions = _multi_dim_range(params, position_dims)
        features = _multi_dim_range(params, feature_dims)
        additive = 0
        if cosine:
            additive = mod(features, 2)
            features = (features - additive) / 2
            additive *= math.pi / 2
            feature_count /= 2
        features -= math.log(math.pi * 2 / position_count) - feature_count / 2
        features *= feature_count ** -0.5
        out = sin(op(positions, exp(features) + additive)) * params.embedding_stddev
        if learned:
            out *= normal_var(params, feature_dims, 1)

    if axial and split:
        feature_dims = []
        position_dims = shape.dims
    if axial:
        out = normal_var(params, [position_dims.pop(0)] + feature_dims, params.embedding_stddev)
        for dim in position_dims:
            out = op(out, normal_var(params, [dim] + feature_dims, params.embedding_stddev))

    if out is None:
        raise ValueError
    if params.shared_position_embedding:
        _EMBEDDINGS[shape] = out

    return out


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

    mid = activate_util(name_extras, _from_feat())
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += activate_util(name_extras, _from_feat())
    return linear_to_features(params, mid, intermediate)
