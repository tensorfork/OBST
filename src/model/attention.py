import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .backend import get_intermediate, linear_from_features, linear_to_features
from .basic import dropout
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum
from ..utils_core import random_name
from ..utils_mtf import anonymize, anonymize_dim, get_attention_dim

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def tf_softmax(x, masked, dim, dim_index, anonymous_dim_index):
    if masked:
        arange = tf.range(0, dim.size)
        msk = tf.reshape(arange, (1, dim.size)) > tf.reshape(arange, (dim.size, 1))
        msk = tf.cast(msk, x.dtype)
        msk = (msk * 3e38) * 2
        shape = [1] * len(x.shape)
        shape[dim_index] = dim.size
        shape[anonymous_dim_index] = dim.size
        msk = tf.reshape(msk, shape)
        x -= msk
    e = tf.exp(x - tf.reduce_max(x, anonymous_dim_index, True))
    return e / tf.reduce_sum(e, anonymous_dim_index, True)


class SoftmaxBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor, dim: mtf.Dimension, masked: bool):
        super().__init__([x, dy], name=random_name("softmax_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.dim = dim
        self.shape: mtf.Shape = x.shape
        self.masked = masked

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        dim_index = self.shape.dims.index(self.dim)
        dim_index = self.shape.dims.index(self.dim)
        anonymous_dim_index = self.shape.dims.index(anonymize_dim(self.dim))
        masked = self.masked
        dim = self.dim

        def slicewise_fn(x, y):
            s = tf_softmax(x, masked, dim, dim_index, anonymous_dim_index)
            dims = ''.join(chr(ord('a') + i) for i in range(len(x.shape)))
            sdims = dims[:anonymous_dim_index] + 'z' + dims[anonymous_dim_index + 1:]
            return s * y - tf.einsum(f"{dims},{sdims},{sdims}->{dims}", s, s, y)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SoftmaxForward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dim: mtf.Dimension, masked: bool):
        super().__init__([x], name=random_name("softmax_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.dim = dim
        self.shape: mtf.Shape = x.shape
        self.masked = masked

    def gradient(self, grad_ys):
        return SoftmaxBackward(self.inputs[0], grad_ys[0], self.dim, self.masked).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        dim_index = self.shape.dims.index(self.dim)
        anonymous_dim_index = self.shape.dims.index(anonymize_dim(self.dim))
        masked = self.masked
        dim = self.dim

        def slicewise_fn(x):
            return tf_softmax(x, masked, dim, dim_index, anonymous_dim_index)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


def attention(args: BlockArgs):
    idx, dim = get_attention_dim(args)
    args.params.attention_idx += 1
    intermediate = get_intermediate(args)
    base = args(dropout(args(activate(args(linear_from_features(args, intermediate))))))
    masked = idx in args.params.masked_attention_dimensions

    key = 0
    if 'embedded' in args or 'context' in args:
        key = linear_to_features(base, intermediate) * dim.size ** -0.5
    if 'embedded' in args or 'positional' in args:
        key += embed(args, [dim] + intermediate)
    val = linear_to_features(base, intermediate)
    qry = linear_to_features(base, intermediate)

    key = anonymize(key, dim)
    val = anonymize(val, dim)

    lgt = einsum([qry, anonymize(key, dim)], reduced_dims=[args.params.key_dim])
    return einsum(SoftmaxForward(lgt, dim, masked).outputs + [val], args.tensor.shape)
