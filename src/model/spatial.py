import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import normal_var, orthogonal_var
from .basic import activated_linear_in, activated_linear_out
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, greater_equal
from ..utils_core import random_name
from ..utils_mtf import (anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked, get_intermediate,
                         linear_shapes)

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


def _softmax_attention(args: BlockArgs, val: mtf.Tensor, *lgt_in: mtf.Tensor) -> mtf.Tensor:
    dim = get_attention_dim(args).dim
    shape = args.tensor.shape
    lgt_in = list(lgt_in)
    lgt_in[0] *= dim.size ** -0.5
    logit_shape = shape - (mtf.Shape(linear_shapes(args).old) - [args.params.head_dim]) + anonymize_dim(dim)
    return einsum(SoftmaxForward(einsum(lgt_in, output_shape=logit_shape), dim, is_masked(args)).outputs + [val], shape)


def attention(args: BlockArgs):
    args.params.attention_idx += 1
    dim = get_attention_dim(args).dim
    base = args(activated_linear_in(args))

    key = 0
    if 'embedded' in args or 'context' in args:
        key = activated_linear_out(base)
    if 'embedded' in args or 'positional' in args:
        key += embed(args, [dim] + args.params.feature_dims)
    return _softmax_attention(args, anonymize(activated_linear_out(base), dim), activated_linear_out(base),
                              anonymize(key, dim))


def spatial_mixing(args: BlockArgs) -> mtf.Tensor:
    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)

    if 'feed_forward' in args:
        args = args(activated_linear_in(args))

    mid = anonymize(args.tensor, dim)
    base = [args.params.head_dim] * ('group' in args)
    old = base + [tmp]
    new = base + [dim]

    inputs = [mid, orthogonal_var(args, old + new)]
    if is_masked(args):
        inputs.append(compare_range(args.params, dim, tmp, greater_equal))

    mid = einsum(inputs, args.tensor.shape)
    if 'multiply_gate' in args:
        if 'tanh' in args:
            mid = mtf.tanh(mid)
        elif 'sigmoid' in args:
            mid = mtf.sigmoid(mid)
        elif 'bias' in args:
            mid += normal_var(args, new, mean=1)
        mid *= args.tensor
    if 'feed_forward' not in args:
        return mid
    return activated_linear_out(args(mid))


def spatial_feed_forward(args: BlockArgs) -> mtf.Tensor:
    base = args(activated_linear_in(args))
    var = orthogonal_var(args, get_intermediate(base) + [anonymize_dim(get_attention_dim(args).dim)])
    return _softmax_attention(args, activated_linear_out(base), var, base.tensor)
