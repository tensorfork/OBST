import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import get_intermediate, normal_var
from ..dataclass import BlockArgs
from ..mtf_wrapper import constant_scalar, einsum, reduce_mean, rsqrt
from ..utils_core import random_name
from ..utils_mtf import dims_from_shape, shape_crossection

tf1 = tf.compat.v1


class GroupNormalizeForward(mtf.Operation):
    def __init__(self, args: BlockArgs, feature_dims: mtf.Shape):
        inputs = [args.tensor]
        if 'scale' in args:
            inputs.append(normal_var(args.params, feature_dims, mean=1))
        if 'shift' in args:
            inputs.append(normal_var(args.params, feature_dims, mean=0))
        super().__init__(inputs, name=random_name("group_normalize_forward"))
        self._outputs = [mtf.Tensor(self, args.tensor.shape, args.tensor.dtype)]
        self.args = args
        self.normed_dim = (feature_dims - [args.params.head_dim]).dims[0]

    def gradient(self, grad_ys):
        return GroupNormalizeBackward(grad_ys, self).outputs

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)

        block_input: mtf.Tensor = self.inputs[0]
        dims = dims_from_shape(block_input)
        feature_dim_index = dims.index(self.normed_dim)

        scale = 'scale' in self.args
        shift = 'shift' in self.args

        if len(self.inputs) > 1:
            feature_map = [mesh_impl.slice_shape([dim])[0] if dim in self.inputs[1].shape.dims else 1
                           for idx, dim in enumerate(block_input.shape.dims)]

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
    def __init__(self, grad_y: typing.List[mtf.Tensor], forward: GroupNormalizeForward):
        super().__init__(grad_y + forward.inputs, name=random_name("group_normalize_backward"))
        self._outputs = [mtf.Tensor(self, inp.shape, inp.dtype) for inp in forward.inputs]
        self.forward = forward

    def lower(self, lowering: mtf.Lowering):
        mesh_impl: mtf.simd_mesh_impl.SimdMeshImpl = lowering.mesh_impl(self)
        _, block_input, *tensors = self.inputs
        block_input: mtf.Tensor = block_input
        dims = dims_from_shape(block_input)

        feature_dim_index = dims.index(self.forward.normed_dim)

        if tensors:
            summed_dims = [idx for idx, dim in enumerate(block_input.shape.dims) if dim not in tensors[0].shape.dims]
            feature_map = [mesh_impl.slice_shape([dim])[0] if dim in tensors[1].shape.dims else 1
                           for idx, dim in enumerate(block_input.shape.dims)]

        scale = 'scale' in self.forward.args
        shift = 'shift' in self.forward.args
        params = self.forward.args.params

        def slicewise_fn(grad_y: tf.Tensor, x: tf.Tensor, *tensors: tf.Tensor):
            tensors = list(tensors)
            size = params.n_embd_per_head
            sum_square = tf.reduce_sum(tf.square(x), feature_dim_index, keepdims=True)
            divisor = tf.math.rsqrt(size * sum_square - tf.square(tf.reduce_sum(x, feature_dim_index, keepdims=True)))
            divisor *= grad_y
            grads = [(3 * sum_square - tf.square(tf.reduce_sum(x, feature_dim_index, keepdims=True) - x))
                     * divisor * size]
            if scale:
                grads[0] *= tf.reshape(tensors.pop(0), feature_map)
                grads.append(tf.reduce_sum(divisor * (x * size - tf.reduce_sum(x, feature_dim_index, keepdims=True)),
                                           summed_dims))
            if shift:
                grads.append(tf.reduce_sum(grad_y, summed_dims))
            return tuple(grads)

        out = mesh_impl.slicewise(slicewise_fn, *(lowering.tensors[inp] for inp in self.inputs))
        for mtf_out, tf_out in zip(self.outputs, out):
            lowering.set_tensor_lowering(mtf_out, tf_out)


def norm(args: BlockArgs) -> mtf.Tensor:
    block_input = args.tensor
    feature_dims = shape_crossection(block_input.shape, args.params.feature_dims + get_intermediate(args))

    if 'group' in args:
        return GroupNormalizeForward(args, feature_dims).outputs[0]

    normalized_shape = block_input.shape - feature_dims
    block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = [rsqrt(1e-6 + einsum([block_input, block_input,
                                  constant_scalar(args.params, normalized_shape.size / block_input.size)],
                                 output_shape=normalized_shape))]
    if 'scale' in args:
        scale.append(normal_var(args.params, feature_dims, mean=1))
    block_input = einsum([block_input] + scale, output_shape=block_input.shape)
    if 'shift' in args:
        block_input += normal_var(args.params, feature_dims, mean=0)
    return block_input
