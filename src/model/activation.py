import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .. import tf_wrapper as tfw
from ..dataclass import BlockArgs
from ..mtf_wrapper import relu as _relu, multiply, einsum, constant, sigmoid as _sigmoid, tanh as _tanh, softplus
from ..utils_core import random_name, scoped

tf1 = tf.compat.v1


class MishForward(mtf.Operation):
    def __init__(self, x: mtf.Tensor):
        super().__init__([x], name=random_name("mish_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return MishBackward(self.inputs[0], grad_ys[0]).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x):
            return tfw.multiply(x, tfw.tanh(tfw.softplus(x)))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class MishBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("mish_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            gte = tfw.tanh(tfw.softplus(x))
            gte += 1. - tfw.square(gte) * x * tfw.sigmoid(x)
            return tfw.multiply(dy, gte)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SiluForward(mtf.Operation):
    def __init__(self, x: mtf.Tensor):
        super().__init__([x], name=random_name("silu_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return SiluBackward(self.inputs[0], grad_ys[0]).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x):
            return tfw.multiply(x, tfw.sigmoid(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SiluBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("silu_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            gte = tfw.sigmoid(x)
            return dy * ((x - 1) * gte + 1)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class LeCunTanhForward(mtf.Operation):
    def __init__(self, x: mtf.Tensor):
        super().__init__([x], name=random_name("lecun_tanh_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return LeCunTanhBackward(self.inputs[0], grad_ys[0]).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x):
            return tfw.tanh(x) + x * 0.1

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class LeCunTanhBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("lecun_tanh_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            return tfw.multiply(dy, tfw.subtract(1.1, tfw.square(tfw.tanh(x))))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SoftsignForward(mtf.Operation):
    def __init__(self, x: mtf.Tensor):
        super().__init__([x], name=random_name("softsign_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def gradient(self, grad_ys):
        return SoftsignBackward(self.inputs[0], grad_ys[0]).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x):
            return x / (1. + tfw.abs(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SoftsignBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("softsign_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            return dy / tfw.square(1. + tfw.abs(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


def _output0(op):
    if not issubclass(op, mtf.Operation):
        raise ValueError

    def _wrapped(args: BlockArgs):
        return op(args.tensor).outputs[0]

    return _wrapped


def _gelu(params, tensor: mtf.Tensor):
    return einsum([tensor, _tanh(einsum([tensor, tensor, tensor, constant(params, 0.044715)],
                                        output_shape=tensor.shape) + tensor * np.sqrt(2 / np.pi)) + 1.0,
                   constant(params, 0.5)], output_shape=tensor.shape)


def gelu(args: BlockArgs):
    return scoped("gelu", _gelu, args.params, args.tensor)


def relu(args: BlockArgs):
    return _relu(args.tensor)


def sigmoid(args: BlockArgs):
    return _sigmoid(args.tensor)


def tanh(args: BlockArgs):
    return _tanh(args.tensor)


def _mtf_mish(tensor: mtf.Tensor):
    return multiply(_tanh(softplus(tensor)), tensor)


def mtf_mish(args: BlockArgs):
    return scoped("mtf_mish", _mtf_mish, args.tensor)


ACTIVATIONS = {'relu': relu,
               'sigmoid': sigmoid,
               'tanh': tanh,
               'gelu': gelu,
               'lecun_tanh': _output0(LeCunTanhForward),
               'silu': _output0(SiluForward),
               'mish': _output0(MishForward),
               "mtf_mish": mtf_mish,
               'softsign': _output0(SoftsignForward)
               }


def activate(args: BlockArgs) -> mtf.Tensor:
    """
    Call activation function on mtf.Tensor.
    """
    for fn_name in args:
        if fn_name not in ACTIVATIONS:
            continue
        return scoped(fn_name, ACTIVATIONS[fn_name], args)
    print(f'No activation function found for "{args.name_extras}". Falling back to identity. '
          f'Known functions: {list(ACTIVATIONS.keys())}')
    return args.tensor
