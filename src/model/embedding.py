import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .backend import normal_var
from ..dataclass import BlockArgs, ModelParameter
from ..mtf_wrapper import einsum, scoped
from ..utils_core import random_name
from ..utils_mtf import DIM_LIST, SHAPE, shape_size

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def _multi_dim_range_tf(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    out, *items = [tf.reshape(tf.range(0, dim.size * size, size),
                              [1] * idx + [dim.size] + [1] * (len(dims) - idx - 1))
                   for idx, (dim, size) in enumerate(zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]])))]
    for i in items:
        out += i
    return tf.cast(out, params.variable_dtype.activation_dtype)


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
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)

        params = self.params
        shape = self.shape

        position_dims: SHAPE = (shape - params.feature_dims) - params.intermediate
        feature_dims = list(set(shape.dims) & set(params.feature_dims + params.intermediate))
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
        out = tf.einsum(f'{position_formula},{feature_formula}->{shape_formula}', positions, features)
        out = tf.math.sin(out) * params.embedding_stddev
        lowering.set_tensor_lowering(self.outputs[0], mesh_impl.import_tf_tensor(self.outputs[0], out))


def _embed(args: BlockArgs, shape: SHAPE) -> mtf.Tensor:
    if isinstance(shape, (list, tuple)):
        shape = mtf.Shape(shape)

    if args.params.shared_position_embedding and shape in args.params.cached_embeddings:
        return args.params.cached_embeddings[shape]

    position_dims: mtf.Shape = (shape - args.params.feature_dims) - args.params.intermediate
    feature_dims = list(set(shape.dims) & set(args.params.feature_dims + args.params.intermediate))

    if 'absolute' in args:
        if 'split' in args:
            out = normal_var(args.params, position_dims, args.params.embedding_stddev)
            out *= normal_var(args.params, feature_dims, args.params.embedding_stddev)
        else:
            out = normal_var(args.params, shape)
    elif 'axial' in args:
        if 'split' in args:
            feature_dims = []
            position_dims = shape.dims
        out = einsum([normal_var(args.params, [dim] + feature_dims, args.params.embedding_stddev)
                      for dim in position_dims], output_shape=shape)
    elif 'relative' in args:
        out = RelativeEmbeddingForward(args.params, shape).outputs[0]
        if 'learned' in args:
            out *= normal_var(args.params, feature_dims, args.params.embedding_stddev)
    else:
        raise ValueError("relative(-learned) or absolute(-split) or axial(-split)")

    if args.params.shared_position_embedding:
        args.params.cached_embeddings[shape] = out

    return out


def embed(args: BlockArgs, shape: SHAPE):
    return scoped('embed', _embed, args, shape)
