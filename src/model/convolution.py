import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .backend import OrthogonalInit, get_attention_dim, get_variable
from ..dataclass import ModelParameter
from ..utils_core import random_name

tf1 = tf.compat.v1


class ConvolutionForward(mtf.Operation):
    def __init__(self, params: ModelParameter, x: mtf.Tensor, dim: mtf.Dimension, kernel_size: int, masked: bool):
        shape: mtf.Shape = x.shape
        batch = shape.dims[0].size
        self.sizes = sizes = [d.size for d in shape]
        space_dims = (shape - params.intermediate - params.feature_dims).dims[1:]
        dim_index = shape.dims.index(dim)
        space_dim_index = space_dims.index(dim)
        features = params.mesh_impl.slice_size(mtf.Shape(params.feature_dims))

        self.weight_size = [features, features]
        self.kwargs = {'stride': 1, 'name': random_name('conv'), 'dilations': 1}

        if len(space_dims) == 1:
            self.kwargs['data_format'] = 'NWC'
            self.weight_size.append(kernel_size)
            self.input_size = [s.size for s in shape if s not in params.feature_dims] + [features]
            self.conv = tf.nn.conv1d
            input2d = self.input_size.copy()
            weight2d = self.weight_size.copy()
            input2d.insert(2, 1)
            weight2d.append(1)

            def back_filter(x, w, dy, **kwargs):
                x = tf.reshape(x, input2d)
                w = tf.reshape(w, weight2d)
                dy = tf.reshape(dy, input2d)
                out = tf1.nn.conv2d_backprop_filter(x, w, dy, **kwargs)
                return tf.reshape(out, self.input_size)

            def back_input(dy, w, **kwargs):
                w = tf.reshape(w, weight2d)
                dy = tf.reshape(dy, input2d)
                out = tf1.nn.conv2d_backprop_input(dy.shape, w, dy, **kwargs)
                return tf.reshape(out, self.input_size)

            self.filter_backprop = back_filter
            self.input_backprop = back_input
        elif space_dim_index == 0:
            self.kwargs['data_format'] = 'NHWC'
            self.weight_size.extend([kernel_size, 1])
            self.input_size = [batch, sizes[1], int(np.prod(sizes[2:len(space_dims)])), features]
            self.conv = tf.nn.conv2d
            self.filter_backprop = tf1.nn.conv2d_backprop_filter
            self.input_backprop = tf.nn.conv2d_transpose
        elif space_dim_index == len(space_dims) - 1:
            self.kwargs['data_format'] = 'NHWC'
            self.weight_size.extend([1, kernel_size])
            self.input_size = [batch, int(np.prod(sizes[1:len(space_dims) - 1])), sizes[len(space_dims)], features]
            self.conv = tf.nn.conv2d
            self.filter_backprop = tf1.nn.conv2d_backprop_filter
            self.input_backprop = tf.nn.conv2d_transpose
        else:
            self.kwargs['data_format'] = 'NDHWC'
            self.weight_size.extend([1, kernel_size, 1])
            self.input_size = [batch, int(np.prod(sizes[1:dim_index])), sizes[dim_index],
                               int(np.prod(sizes[dim_index + 1:len(space_dims)])), features]
            self.conv = tf.nn.conv3d
            self.filter_backprop = tf1.nn.conv3d_backprop_filter_v2
            self.input_backprop = tf.nn.conv3d_transpose
        self.kwargs['padding'] = 'SAME'
        if masked:
            self.kwargs['padding'] = [[w - 1, 0] for w in self.weight_size[2:]]

        fan_in = [mtf.Dimension(chr(i + ord('a')), w) for i, w in enumerate(self.weight_size[1:]) if w != 1]
        mtf_weight_size = params.feature_dims
        mtf_weight_size.extend(fan_in)
        super().__init__([x, get_variable(params, mtf_weight_size, OrthogonalInit(params, mtf_weight_size, fan_in))],
                         name=random_name("conv_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.params = params

    def gradient(self, grad_ys):
        return ConvolutionFilterBackward(self).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, w):
            x = tf.reshape(x, self.input_size)
            w = tf.reshape(w, self.weight_size)
            out = self.conv(x, w, **self.kwargs)
            return tf.reshape(out, self.sizes)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class ConvolutionFilterBackward(mtf.Operation):
    def __init__(self, conv: ConvolutionForward):
        super().__init__(conv.inputs + conv.outputs, name=random_name("conv_backward"))
        self._outputs = conv.inputs
        self.conv = conv

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        conv = self.conv

        def slicewise_fn(x, w, dy):
            x = tf.reshape(x, conv.input_size)
            w = tf.reshape(w, conv.weight_size)
            dy = tf.reshape(dy, conv.input_size)
            back_filter = conv.filter_backprop(x, w, dy, **conv.kwargs)
            back_input = conv.input_backprop(dy, w, **conv.kwargs)
            back_filter = tf.reshape(back_filter, conv.input_size)
            back_input = tf.reshape(back_input, conv.input_size)
            return back_input, back_filter

        dx, dw = mesh_impl.slicewise(slicewise_fn, *[lowering.tensors[self.inputs[i]] for i in range(3)])
        lowering.set_tensor_lowering(self.outputs[0], dx)
        lowering.set_tensor_lowering(self.outputs[1], dw)


def convolution(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    idx, dim = get_attention_dim(params, block_input)
    convolution_size = 16
    if len(name_extras) > 0 and name_extras[-1].isdigit():
        convolution_size = int(name_extras[-1])
    return ConvolutionForward(params, block_input, dim, convolution_size,
                              idx in params.masked_attention_dimensions).outputs[0]
