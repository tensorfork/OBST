import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .activation import activate_util
from .backend import get_variable, linear_from_features, linear_to_features, normal_var, \
    orthogonal_var
from ..dataclass import ModelParameter
from ..mtf_wrapper import (add_n, constant_scalar, dropout as utils_dropout, einsum, exp, mod, mtf_range, reduce_mean,
                           rsqrt, sigmoid, sin)
from ..utils_core import random_name
from ..utils_mtf import (DIM_LIST, SHAPE, anonymize_dim, deduplicate, dims_from_shape, guarantee_const, new_dim,
                         shape_size)

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


class GroupFeedForwardForward(mtf.Operation):
    def __init__(self, params: ModelParameter, block_input: mtf.Tensor):
        self.params = params
        shape = [params.key_dim, anonymize_dim(params.key_dim, params.n_embd_per_head * params.group_linear_factor)]
        new_key = new_dim(params.key_dim, params.n_embd * params.group_linear_factor)
        super().__init__([block_input, orthogonal_var(params, shape)], name=random_name("gff_forward"))
        out_shape = block_input.shape
        if params.key_dim in out_shape.dims:
            self.input_dim, self.output_dim = params.key_dim, new_key
        else:
            self.input_dim, self.output_dim = new_key, params.key_dim
        self._outputs = [mtf.Tensor(self, out_shape - self.input_dim + self.output_dim, block_input.dtype)]

    def gradient(self, grad_ys):
        return GroupFeedForwardBackward(grad_ys, self.params, self)

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)
        block_input: mtf.Tensor = self.inputs[0]
        block_output: mtf.Tensor = self.outputs[0]
        dims = deduplicate(block_input.shape.dims + self.input_dim + self.output_dim)
        input_formula = ','.join(''.join(chr(dims.index(d)) for d in inp.shape) for inp in self.inputs)
        output_formula = ''.join(chr(dims.index(d)) for d in block_output.shape)
        einsum_formula = f'{input_formula}->{output_formula}'

        def slicewise_fn(left, right):
            return tf.einsum(einsum_formula, left, right)

        y = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        lowering.set_tensor_lowering(block_output, y)


class GroupFeedForwardBackward(mtf.Operation):
    def __init__(self, grad_ys: mtf.Tensor, params: ModelParameter, forward: GroupFeedForwardForward):
        super().__init__([grad_ys] + forward.inputs, name=random_name("gff_backward"))
        self.params = params
        self.input_dim, self.output_dim = forward.input_dim, forward.output_dim
        self._outputs = [mtf.Tensor(self, inp.shape, inp.dtype) for inp in forward.inputs]

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)
        grad_ys: mtf.Tensor = self.inputs[0]
        block_input: mtf.Tensor = self.inputs[1]
        var: mtf.Tensor = self.inputs[2]

        dims = deduplicate(block_input.shape.dims + self.input_dim + self.output_dim)

        input_formula0 = ','.join(''.join(chr(dims.index(d)) for d in inp.shape) for inp in [grad_ys, var])
        output_formula0 = ''.join(chr(dims.index(d)) for d in block_input.shape)
        einsum_formula0 = f'{input_formula0}->{output_formula0}'
        input_formula1 = ','.join(''.join(chr(dims.index(d)) for d in inp.shape) for inp in [grad_ys, block_input])
        output_formula1 = ''.join(chr(dims.index(d)) for d in var.shape)
        einsum_formula1 = f'{input_formula1}->{output_formula1}'

        def slicewise_fn(grad, inp, tf_var):
            return [tf.einsum(einsum_formula0, grad, tf_var), tf.einsum(einsum_formula1, grad, inp)]

        y = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        lowering.set_tensor_lowering(self.outputs[0], y[0])
        lowering.set_tensor_lowering(self.outputs[1], y[1])


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

        params = self.params

        if len(self.inputs) > 1:
            feature_map = [mesh_impl.slice_shape([dim])[0] if dim in block_input.shape.dims else 1
                           for idx, dim in enumerate(self.inputs[1].shape.dims)]

        def slicewise_fn(*tensors: tf.Tensor):
            tensors = list(tensors)
            x = tensors.pop(0)
            shape = x.shape.as_list()
            contract_dims = [feature_dim_index + 1]
            x = tf.reshape(x,
                           shape[:feature_dim_index] + [params.n_head, params.n_embd_per_head] +
                           shape[feature_dim_index + 1:])
            x -= tf.reduce_mean(x, contract_dims)
            x /= tf.reduce_mean(tf.square(x), contract_dims)
            x = tf.reshape(x, shape)
            if scale:
                x *= tf.reshape(tensors.pop(0), feature_map)
            if shift:
                x += tf.reshape(tensors.pop(0), feature_map)
            return x

        y = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        lowering.set_tensor_lowering(self.outputs[0], y)


class GroupNormalizeBackward(mtf.Operation):
    def __init__(self, grad_y: mtf.Tensor, params: ModelParameter, name_extras: typing.List[str],
                 tensors: typing.List[mtf.Tensor]):
        super().__init__([grad_y] + tensors, name=random_name("group_normalize_backward"))
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
            feature_map = [mesh_impl.slice_shape([dim])[0] if dim in block_input.shape.dims else 1
                           for idx, dim in enumerate(tensors[0].shape.dims)]
            summed_dims = [idx for idx, dim in enumerate(block_input.shape.dims) if dim not in tensors[0].shape.dims]

        name_extras = self.name_extras
        scale = 'scale' in name_extras
        shift = 'shift' in name_extras

        params = self.params

        def slicewise_fn(grad_y: tf.Tensor, x: tf.Tensor, *tensors: tf.Tensor):
            tensors = list(tensors)
            shape = x.shape.as_list()
            contract_dims = [feature_dim_index + 1]
            size = params.n_embd_per_head
            x = tf.reshape(x, shape[:feature_dim_index] + [params.n_head, size] + shape[feature_dim_index + 1:])
            sum_square = tf.reduce_sum(tf.square(x), contract_dims)
            divisor = tf.math.rsqrt(size * sum_square - tf.square(tf.reduce_sum(x, contract_dims))) * grad_y
            grads = [divisor * size * (3 * sum_square - tf.square(tf.reduce_sum(x, contract_dims) - x))]
            if scale:
                grads[0] *= tf.reshape(tensors.pop(0), feature_map)
                grads.append(tf.reduce_sum(divisor * (x * size - tf.reduce_sum(x, contract_dims)), summed_dims))
            if shift:
                grads.append(tf.reduce_sum(grad_y, summed_dims))
            return tuple(grads)

        out = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        for mtf_out, tf_out in zip(self.outputs, out):
            lowering.set_tensor_lowering(mtf_out, tf_out)


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
    return guarantee_const(add_n([mtf_range(params.mesh, dim, params.variable_dtype.activation_dtype) * size
                                  for dim, size in zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]]))]))


_EMBEDDINGS = {}


def embed(params: ModelParameter, shape: SHAPE) -> mtf.Tensor:
    if isinstance(shape, list):
        shape = mtf.Shape(shape)
    position_dims: SHAPE = (shape - params.feature_dims) - params.intermediate

    feature_dims = list(set(shape.dims).union(set(params.feature_dims + params.intermediate)))
    position_count = shape_size(position_dims)
    feature_count = shape_size(feature_dims)

    op = mtf.add if 'additive' in params.position_embedding else mtf.multiply
    split = 'split' in params.position_embedding
    absolute = 'absolute' in params.position_embedding
    relative = 'relative' in params.position_embedding
    axial = 'axial' in params.position_embedding
    learned = 'learned' in params.position_embedding
    cosine = 'cosine' in params.position_embedding

    out = None

    if params.shared_position_embedding and shape in _EMBEDDINGS:
        return _EMBEDDINGS[shape]

    if split:
        if absolute:
            out = op(normal_var(params, position_dims, params.embedding_stddev),
                     normal_var(params, feature_dims, params.embedding_stddev))
        elif relative and learned:
            positions = _multi_dim_range(params, position_dims)
            out = op(positions, normal_var(params, feature_dims, 1 / position_count))
        elif axial:
            feature_dims = []
            position_dims = shape.dims
    elif absolute:
        out = normal_var(params, shape, params.embedding_stddev)
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
        out = guarantee_const(sin(op(positions, exp(features) + additive)) * params.embedding_stddev)
        if learned:
            out *= normal_var(params, feature_dims, 1)
    if axial:
        out = normal_var(params, [position_dims.pop(0)] + feature_dims, params.embedding_stddev)
        for dim in position_dims:
            out = op(out, normal_var(params, [dim] + feature_dims, params.embedding_stddev))

    if out is None:
        raise ValueError
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
    if 'group' in name_extras:
        def _from_feat():
            return GroupFeedForwardForward(params, block_input)

        def _out(x):
            return GroupFeedForwardForward(params, x)
    else:
        def _from_feat():
            return linear_from_features(params, block_input, params.intermediate)

        def _out(x):
            return linear_to_features(params, x, params.intermediate)

    mid = dropout(params, activate_util(name_extras, _from_feat()), name_extras)
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += activate_util(name_extras, _from_feat())
    return _out(mid)
