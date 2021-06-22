import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .basic import activated_linear_in, activated_linear_out
from .embedding import embed
from ..dataclass import BlockArgs
from ..mtf_wrapper import einsum, greater_equal, multiply
from ..utils_core import random_name
from ..utils_mtf import (anonymize, anonymize_dim, compare_range, get_attention_dim, is_masked, linear_shapes)

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


def _masked_map(args: BlockArgs):
    dim = get_attention_dim(args).dim
    tmp = anonymize_dim(dim)
    bias = embed(args, [args.params.head_dim, dim, tmp])
    return bias, compare_range(args.params, dim, tmp, greater_equal) if is_masked(args) else 1


def attention(args: BlockArgs):
    args.params.attention_idx += 1
    if "dot_product" in args or "input_as_value" not in args:
        base = args(activated_linear_in(args))

    dim = get_attention_dim(args).dim
    shape = args.tensor.shape

    logit = 0
    val = 0
    key = 0
    if 'dot_product' in args:
        if 'embedded' in args or 'context' in args:
            key = activated_linear_out(base)
        if 'embedded' in args or 'positional' in args:
            key += embed(args, [dim] + args.params.feature_dims)
        qry = activated_linear_out(base)
        qry *= dim.size ** -0.5
        logit_shape = shape - (mtf.Shape(linear_shapes(args).old) - [args.params.head_dim]) + anonymize_dim(dim)
        logit = einsum([qry, anonymize(key, dim)], output_shape=logit_shape)
        if "shared_key_value" in args:
            val = key
    if 'biased_softmax' in args:
        logit += multiply(*_masked_map(args))
    if logit != 0:
        logit = SoftmaxForward(logit, dim, is_masked(args)).outputs[0]
    if 'biased_attention_map' in args:
        logit += multiply(*_masked_map(args))
    logit = [logit] * (logit != 0)
    if 'scale_attention_map':
        logit.extend(_masked_map(args))
    if val == 0:
        val = anonymize(args.tensor if "input_as_value" else activated_linear_out(base), dim)
    if not logit:
        raise UserWarning(f"WARNING: There is no spatial mixing happening with the following attention parameters: "
                          f"{args.name_extras}.")
    return einsum(logit + [val], shape)
