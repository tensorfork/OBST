import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .activation import activate_util
from .backend import get_intermediate, get_variable, linear_from_features, linear_to_features, normal_var
from ..dataclass import ModelParameter
from ..mtf_wrapper import (add_n, constant_scalar, dropout as utils_dropout, einsum, mtf_range, reduce_mean,
                           rsqrt, sigmoid)
from ..utils_core import random_name
from ..utils_mtf import DIM_LIST, SHAPE, dims_from_shape, shape_size

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


class GroupNormalizeForward(mtf.Operation):
    def __init__(self, params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
        inputs = [block_input]
        if 'scale' in name_extras:
            inputs.append(normal_var(params, params.feature_dims, mean=1))
        if 'shift' in name_extras:
            inputs.append(normal_var(params, params.feature_dims, mean=0))
        super().__init__(inputs, name=random_name("group_normalize_forward"))
        self._outputs = [mtf.Tensor(self, block_input.shape, block_input.dtype)]
        self.name_extras = name_extras
        self.params = params

    def gradient(self, grad_ys):
        return GroupNormalizeBackward(grad_ys, self.params, self.name_extras, self.inputs).outputs

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)

        block_input: mtf.Tensor = self.inputs[0]
        dims = dims_from_shape(block_input)
        feature_dim_index = dims.index(self.params.key_dim)

        name_extras = self.name_extras
        scale = 'scale' in name_extras
        shift = 'shift' in name_extras

        if len(self.inputs) > 1:
            feature_map = [mesh_impl.slice_shape([dim])[0] if dim in block_input.shape.dims else 1
                           for idx, dim in enumerate(self.inputs[1].shape.dims)]

        def slicewise_fn(*tensors: tf.Tensor):
            tensors = list(tensors)
            x = tensors.pop(0)
            x -= tf.reduce_mean(x, feature_dim_index, keepdims=True)
            x /= tf.reduce_mean(tf.square(x), feature_dim_index, keepdims=True)
            if scale:
                x *= tf.reshape(tensors.pop(0), feature_map)
            if shift:
                x += tf.reshape(tensors.pop(0), feature_map)
            return x

        y = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        lowering.set_tensor_lowering(self.outputs[0], y)


class GroupNormalizeBackward(mtf.Operation):
    def __init__(self, grad_y: typing.List[mtf.Tensor], params: ModelParameter, name_extras: typing.List[str],
                 tensors: typing.List[mtf.Tensor]):
        super().__init__(grad_y + tensors, name=random_name("group_normalize_backward"))
        self._outputs = [mtf.Tensor(self, inp.shape, inp.dtype) for inp in tensors]
        self.name_extras = name_extras
        self.params = params

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)
        _, block_input, *tensors = self.inputs
        block_input: mtf.Tensor = block_input
        dims = dims_from_shape(block_input)
        feature_dim_index = dims.index(self.params.key_dim)

        if tensors:
            summed_dims = [idx for idx, dim in enumerate(block_input.shape.dims) if dim not in tensors[0].shape.dims]

        name_extras = self.name_extras
        scale = 'scale' in name_extras
        shift = 'shift' in name_extras

        params = self.params

        def slicewise_fn(grad_y: tf.Tensor, x: tf.Tensor, *tensors: tf.Tensor):
            tensors = list(tensors)
            size = params.n_embd_per_head
            sum_square = tf.reduce_sum(tf.square(x), feature_dim_index, keepdims=True)
            divisor = tf.math.rsqrt(size * sum_square - tf.square(tf.reduce_sum(x, feature_dim_index, keepdims=True)))
            divisor *= grad_y
            grads = [(3 * sum_square - tf.square(tf.reduce_sum(x, feature_dim_index, keepdims=True) - x))
                     * divisor * size]
            if scale:
                grads[0] *= tensors.pop(0)
                grads.append(tf.reduce_sum(divisor * (x * size - tf.reduce_sum(x, feature_dim_index, keepdims=True)),
                                           summed_dims))
            if shift:
                grads.append(tf.reduce_sum(grad_y, summed_dims))
            return tuple(grads)

        out = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        for mtf_out, tf_out in zip(self.outputs, out):
            lowering.set_tensor_lowering(mtf_out, tf_out)


class RelativeEmbeddingForward(mtf.Operation):
    def __init__(self, params: ModelParameter, shape: SHAPE):
        super().__init__([], params.mesh, name=random_name("rel_embed"))
        if isinstance(shape, list):
            shape = mtf.Shape(shape)
        self.params = params
        self.shape = shape
        self._outputs = [mtf.Tensor(self, shape, params.variable_dtype.activation_dtype)]

    def has_gradient(self):
        return False

    def lower(self, lowering: mtf.Lowering):
        params = self.params
        shape = self.shape

        position_dims: SHAPE = (shape - params.feature_dims) - params.intermediate
        feature_dims = list(set(shape.dims).union(set(params.feature_dims + params.intermediate)))
        position_count = shape_size(position_dims)

        cosine = 'cosine' in params.position_embedding

        shape_formula = ''.join(chr(ord('a') + i) for i in range(shape.ndims))
        position_formula = ''.join(shape_formula[shape.dims.index(d)] for d in position_dims)
        feature_formula = ''.join(shape_formula[shape.dims.index(d)] for d in feature_dims)

        positions = _multi_dim_range_tf(params, position_dims)
        features = _multi_dim_range_tf(params, feature_dims)
        additive = 0
        feature_count = shape_size(feature_dims)

        if cosine:
            additive = tf.math.mod(features, 2)
            features = (features - additive) / 2
            additive *= math.pi
            feature_count /= 2

        features -= math.log(math.pi * 2 / position_count) - feature_count / 2
        features *= feature_count ** 0.5
        features = tf.exp(features) + additive
        out = tf.einsum(f'{position_formula},{feature_formula},{shape_formula}', positions, features)
        out = tf.math.sin(out) * params.embedding_stddev

        lowering.set_tensor_lowering(self.outputs[0], out)


def norm(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    if 'group' in name_extras:
        return GroupNormalizeForward(params, block_input, name_extras).outputs[0]

    normalized_shape = block_input.shape - [params.key_dim]
    normalized_shape = normalized_shape - [params.head_dim]
    block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = [rsqrt(1e-6 + einsum([block_input, block_input,
                                  constant_scalar(params, normalized_shape.size / block_input.size)],
                                 output_shape=normalized_shape))]
    if 'scale' in name_extras:
        scale.append(normal_var(params, params.feature_dims, mean=1))
    block_input = mtf.einsum([block_input] + scale, output_shape=block_input.shape)
    if 'shift' in name_extras:
        block_input += normal_var(params, params.feature_dims, mean=0)
    return block_input


def rezero(params, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return block_input * get_variable(params, [], tf.constant_initializer(0))


def _multi_dim_range(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    return add_n([mtf_range(params.mesh, dim, params.variable_dtype.activation_dtype) * size
                  for dim, size in zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]]))])


def _multi_dim_range_tf(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    return add_n([tf.range(0, dim.size * size, size, dtype=params.variable_dtype.activation_dtype)
                  for dim, size in zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]]))])


_EMBEDDINGS = {}


def embed(params: ModelParameter, shape: SHAPE, name_extras: typing.Union[typing.List[str], str]) -> mtf.Tensor:
    if isinstance(shape, (list, tuple)):
        shape = mtf.Shape(shape)

    if params.shared_position_embedding and shape in _EMBEDDINGS:
        return _EMBEDDINGS[shape]

    position_dims: mtf.Shape = (shape - params.feature_dims) - params.intermediate
    feature_dims = list(set(shape.dims).union(set(params.feature_dims + params.intermediate)))

    if 'absolute' in name_extras:
        if 'split' in name_extras:
            out = normal_var(params, position_dims, params.embedding_stddev)
            out *= normal_var(params, feature_dims, params.embedding_stddev)
        else:
            out = normal_var(params, shape)
    elif 'axial' in name_extras:
        if 'split' in name_extras:
            feature_dims = []
            position_dims = shape.dims
        out = einsum([normal_var(params, [dim] + feature_dims, params.embedding_stddev) for dim in position_dims],
                     output_shape=shape)
    elif 'relative' in name_extras:
        out = RelativeEmbeddingForward(params, shape).outputs[0]
        if 'learned' in name_extras:
            out *= normal_var(params, feature_dims, params.embedding_stddev)
    else:
        raise ValueError("relative(-learned) or absolute(-split) or axial(-split)")

    if params.shared_position_embedding:
        _EMBEDDINGS[shape] = out

    return out


def dropout(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    keep = 1
    for extra in name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(block_input, keep)


def feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    intermediate = get_intermediate(params, name_extras)

    def _from_feat():
        return linear_from_features(params, block_input, intermediate)

    mid = dropout(params, activate_util(name_extras, _from_feat()), name_extras)
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += activate_util(name_extras, _from_feat())
    return linear_to_features(params, mid, intermediate)
