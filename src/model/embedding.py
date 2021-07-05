import math

import mesh_tensorflow as mtf
import numpy as np

from .backend import normal_var, orthogonal_var
from .. import tf_wrapper as tfw
from ..dataclass import BlockArgs, ModelParameter
from ..mtf_wrapper import einsum, reshape, multiply
from ..utils_core import random_name, scoped
from ..utils_mtf import DIM_LIST, SHAPE, linear_shapes, shape_size


def _multi_dim_range_tf(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    out, *items = [tfw.reshape(tfw.tf_range(0, dim.size * size, size),
                               [1] * idx + [dim.size] + [1] * (len(dims) - idx - 1))
                   for idx, (dim, size) in enumerate(zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]])))]
    for i in items:
        out = tfw.add(out, i)
    return tfw.cast(out, params.variable_dtype.activation_dtype)


class RelativeEmbeddingForward(mtf.Operation):
    def __init__(self, args: BlockArgs, shape: SHAPE):
        super().__init__([], args.params.mesh, name=random_name("rel_embed"))
        if isinstance(shape, list):
            shape = mtf.Shape(shape)
        self.args = args
        self.shape = shape
        self._outputs = [mtf.Tensor(self, shape, args.params.variable_dtype.activation_dtype)]

    def has_gradient(self):
        return False

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)

        params = self.args.params
        shape = self.shape

        position_dims: SHAPE = (shape - params.feature_dims) - params.intermediate
        feature_dims = linear_shapes(self.args).old
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
            additive = tfw.mod(features, 2)
            features = tfw.divide(tfw.subtract(features, additive), 2)
            additive = tfw.multiply(additive, math.pi)
            feature_count /= 2

        features = tfw.multiply(features, 4 / feature_count)
        features = tfw.subtract(features, math.log(position_count / 2 / math.pi))
        features = tfw.add(tfw.exp(features), additive)
        out = tfw.einsum(f'{position_formula},{feature_formula}->{shape_formula}', positions, features)
        out = multiply(tfw.sin(out), params.embedding_stddev)
        lowering.set_tensor_lowering(self.outputs[0], mesh_impl.import_tf_tensor(self.outputs[0], out))


def _embed_var(args: BlockArgs, shape: SHAPE) -> mtf.Tensor:
    if 'orthogonal' in args:
        return orthogonal_var(args, shape)
    return normal_var(args, shape, args.params.embedding_stddev)


def _embed(args: BlockArgs, shape: SHAPE) -> mtf.Tensor:
    if isinstance(shape, (list, tuple)):
        shape = mtf.Shape(shape)

    variables = []
    position_dims: mtf.Shape = (shape - args.params.feature_dims) - args.params.intermediate
    feature_dims = linear_shapes(args).old

    if 'absolute' in args:
        out = _embed_var(args, shape)
    elif 'axial' in args:
        splits = 2
        for a in args:
            if a.isdigit():
                splits = int(a)
                break
        tmp_dims = []
        variables = []

        def _new_part(size: int):
            tmp = mtf.Dimension(f'_{len(tmp_dims)}', size)
            tmp_dims.append(tmp)
            variables.append(_embed_var(args, [tmp] + feature_dims))

        for dim in position_dims:
            base = int(dim.size ** (1 / splits))
            while dim.size % base != 0:
                base -= 1
            final = dim.size // base ** (splits - 1)
            _new_part(final)
            for i in range(1, splits):
                _new_part(base)
        out = reshape(einsum(variables, output_shape=tmp_dims + feature_dims), shape)

    elif 'relative' in args:
        out = RelativeEmbeddingForward(args, shape).outputs[0]
        if 'learned' in args:
            out = multiply(out, _embed_var(args, feature_dims))
    else:
        raise ValueError("The following embeddings are supported:"
                         " relative(-learned) or absolute(-split) or axial(-split) are supported")

    return out


def embed(args: BlockArgs, shape: SHAPE):
    return scoped('embed', _embed, args, shape)
