import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .backend import normal_var, orthogonal_var
from .. import tf_wrapper as tfw
from ..dataclass import BlockArgs, ModelParameter
from ..mtf_wrapper import einsum, reshape, multiply, zeros_like
from ..utils_core import random_name, scoped
from ..utils_mtf import DIM_LIST, SHAPE, linear_shapes, shape_size


def _multi_dim_range_tf(params: ModelParameter, dims: DIM_LIST) -> mtf.Tensor:
    out, *items = [tfw.reshape(tfw.tf_range(0, dim.size * size, size),
                               [1] * idx + [dim.size] + [1] * (len(dims) - idx - 1))
                   for idx, (dim, size) in enumerate(zip(dims, np.cumprod([1] + [d.size for d in dims[:-1]])))]
    for i in items:
        out += i
    return tfw.cast(out, params.variable_dtype.activation_dtype)


def squeeze(tensor: tf.Tensor, removed_dims: typing.List[int]):
    shape = tensor.shape.as_list()
    for i, d in enumerate(sorted(removed_dims)):
        del shape[d - i]
    return tf.reshape(tensor, shape)


def unsqueeze(tensor: tf.Tensor, added_dims: typing.List[int]):
    shape = tensor.shape.as_list()
    for d in sorted(added_dims):
        shape.insert(d, 1)
    return tf.reshape(tensor, shape)


class ScatterAdd(mtf.Operation):
    """Assign to one or more variables."""

    def __init__(self, out: mtf.Tensor, indices: mtf.Tensor, gradient: mtf.Tensor,
                 squeeze_dims: typing.Optional[SHAPE]):
        super().__init__([out, indices, gradient], out.mesh, random_name("sparse_assign"))
        if isinstance(squeeze_dims, mtf.Shape):
            squeeze_dims = squeeze_dims.dims
        if squeeze_dims is None:
            squeeze_dims = []
        self.squeeze_dims = squeeze_dims
        self.index_dims = [indices.shape.dims.index(dim) for dim in squeeze_dims if dim in indices.shape.dims]
        self.embed_dims = [out.shape.dims.index(dim) for dim in squeeze_dims if dim in out.shape.dims]
        self.grad_dims = [gradient.shape.dims.index(dim) for dim in squeeze_dims if dim in gradient.shape.dims]

        self.indices = indices
        self.grad = gradient
        self._outputs = [mtf.Tensor(self, out.shape, out.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        flattened_dims = 0

        def assign_fn(val: tf.Tensor, indices: tf.Tensor, gradient: tf.Tensor) -> tf.Tensor:
            # Val: [1 (Heads), Keys, Features]
            # Indices: [Batch, Sequence, 1 (Heads), 1 (Keys)]
            # Gradient: [Batch, Sequence, 1 (Heads), Features]
            shape = val.shape
            val = squeeze(val, self.embed_dims)
            indices = squeeze(indices, self.index_dims)
            gradient = squeeze(gradient, self.grad_dims)
            indices = tf.reshape(indices, indices.shape.as_list() + [1])
            val = tf.reshape(val, val.shape.as_list()[:-flattened_dims] + [-1])
            gradient = tf.cast(tf.reshape(gradient, gradient.shape.as_list()[:-flattened_dims] + [-1]), val.dtype)
            return tf.reshape(tf.tensor_scatter_nd_add(val, indices, gradient), shape)

        out, indices, gradients = self.inputs
        for flattened_dims, (dim0, dim1) in enumerate(zip((out.shape - self.squeeze_dims).dims[::-1],
                                                          (gradients.shape - self.squeeze_dims).dims[::-1]), 0):
            if dim0 != dim1:
                break
        flattened_dims = max(flattened_dims, 1)
        y = mesh_impl.slicewise(assign_fn, lowering.tensors[out], lowering.tensors[indices],
                                lowering.tensors[gradients])
        lowering.set_tensor_lowering(self.outputs[0], y)


def scatter_add(out: mtf.Tensor, indices: mtf.Tensor, gradient: mtf.Tensor, squeeze_dims: typing.Optional[SHAPE] = None
                ) -> mtf.Tensor:
    return ScatterAdd(out, indices, gradient, squeeze_dims).outputs[0]


class Gather(mtf.Operation):
    def __init__(self, args: BlockArgs, embedding: mtf.Tensor, squeeze_dims: typing.Optional[SHAPE]):
        super().__init__([args.tensor, embedding], args.params.mesh, name=random_name("gather"))
        if isinstance(squeeze_dims, mtf.Shape):
            squeeze_dims = squeeze_dims.dims
        if squeeze_dims is None:
            squeeze_dims = []
        self.squeeze_dims = squeeze_dims
        self.squeezed_index_dims = [args.tensor.shape.dims.index(dim) for dim in squeeze_dims
                                    if dim in args.tensor.shape.dims]
        self.squeezed_embed_dims = [embedding.shape.dims.index(dim) for dim in squeeze_dims
                                    if dim in embedding.shape.dims]
        out_shape = args.tensor.shape - squeeze_dims + embedding.shape.dims[1:]
        self.args = args
        self.unsqueezed_dims = [out_shape.dims.index(dim) for dim in squeeze_dims if dim in out_shape.dims]
        self._outputs = [mtf.Tensor(self, out_shape,
                                    args.params.variable_dtype.activation_dtype)]

    def gradient(self, grad_ys: typing.List[mtf.Tensor]) -> typing.Tuple[None, mtf.Tensor]:
        indices, embedding = self.inputs
        return None, scatter_add(zeros_like(embedding), indices, grad_ys[0], self.squeeze_dims)

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)

        indices, embeddings = self.inputs

        def slicewise_fn(idx: tf.Tensor, embd: tf.Tensor) -> tf.Tensor:
            idx = squeeze(idx, self.squeezed_index_dims)
            embd = squeeze(embd, self.squeezed_embed_dims)
            out = tf.gather(embd, idx, axis=0)
            return unsqueeze(out, self.unsqueezed_dims)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[indices], lowering.tensors[embeddings])
        lowering.set_tensor_lowering(self.outputs[0], y)


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
            features = (features - additive) / 2
            additive = additive * math.pi
            feature_count /= 2

        features += 4 / feature_count
        features -= math.log(position_count / 2 / math.pi)
        features = tfw.exp(features) + additive
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


def embed(args: BlockArgs, shape: SHAPE) -> mtf.Tensor:
    return scoped('embed', _embed, args, shape)


def gather_embed(args: BlockArgs, shape: SHAPE, squeezed_dims: typing.Optional[SHAPE] = None) -> mtf.Tensor:
    return Gather(args, scoped("gather", embed, args, shape), squeezed_dims).outputs[0]
