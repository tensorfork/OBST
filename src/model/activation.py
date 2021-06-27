import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import BlockArgs
from ..mtf_wrapper import relu, gelu, sigmoid, tanh, scoped
from ..utils_core import random_name

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
            return x * tf.tanh(tf.math.softplus(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class MishBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("mish_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            gte = tf.math.tanh(tf.math.softplus(x))
            return dy * (gte + (1 - tf.math.square(gte)) * x * tf.math.sigmoid(x))

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
            return x * tf.math.sigmoid(x)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SiluBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("silu_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            gte = tf.math.sigmoid(x)
            return [dy * (x * gte + (1 - gte))]

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
            return tf.math.tanh(x) + x * 0.1

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class LeCunTanhBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("lecun_tanh_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            return dy * (1.1 - tf.math.square(tf.math.tanh(x)))

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
            return x / (1 + tf.math.abs(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class SoftsignBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor):
        super().__init__([x, dy], name=random_name("softsign_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def slicewise_fn(x, dy):
            return dy / tf.math.square(1 + tf.math.abs(x))

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


def _output0(op):
    if not issubclass(op, mtf.Operation):
        raise ValueError

    def _wrapped(*args, **kwargs):
        return op(*args, **kwargs).outputs[0]

    return _wrapped


ACTIVATIONS = {'relu': relu,
               'sigmoid': sigmoid,
               'tanh': tanh,
               'gelu': gelu,
               'lecun_tanh': _output0(LeCunTanhForward),
               'silu': _output0(SiluForward),
               'mish': _output0(MishForward),
               'softsign': _output0(SoftsignForward)
               }


def activate(args: BlockArgs) -> mtf.Tensor:
    """
    Call activation function on mtf.Tensor.
    """
    for fn_name in args:
        if fn_name not in ACTIVATIONS:
            continue
        return scoped(fn_name, ACTIVATIONS[fn_name], args.tensor)
    print(f'No activation function found for "{args.name_extras}". Falling back to identity. '
          f'Known functions: {list(ACTIVATIONS.keys())}')
    return args.tensor
