import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .frontend import block_part_fn
from ..utils_core import random_name
from ..dataclass import ModelParameter

tf1 = tf.compat.v1


class MomentumOperation(mtf.Operation):
    """Operation to implement custom gradients.

    See comments on custom_gradient() below.
    """

    def __init__(self, params, block_config, x, x_backwards, v, v_backwards, index):
        graph: mtf.Graph = x.graph
        prev_ops = len(graph.operations)
        new_v = v * params.momentumnet_alpha
        fx = block_part_fn(params, block_config, x, index)
        new_v += fx * (1 - params.momentumnet_alpha)
        new_x = x + new_v
        fn_outputs = [new_x, x_backwards, new_v, v_backwards]
        forward_operations = graph.operations[prev_ops:]
        new_outputs = set()
        new_inputs = set()
        for op in forward_operations:
            new_inputs.update(set(op.inputs))
            if not isinstance(op, mtf.Variable):
                new_outputs.update(set(op.outputs))
        explicit_inputs = [x, x_backwards, v, v_backwards]
        variables = [t for t in list(new_inputs - new_outputs - set(explicit_inputs)) if t.dtype.is_floating]
        super(MomentumOperation, self).__init__(explicit_inputs + variables + fn_outputs, params.mesh,
                                                random_name("custom_gradient"))
        # Make sure no one uses the internals of this function, since the gradients
        #  will probably not work correctly.
        for t in new_outputs - set(fn_outputs):
            t.usable = False

        self._graph: mtf.Graph = x.graph
        self._x: mtf.Tensor = x
        self.params: ModelParameter = params
        self._v: mtf.Tensor = v
        self._fx: mtf.Tensor = fx
        self._variables: typing.List[mtf.Variable] = variables
        self._fn_outputs: typing.List[mtf.Tensor] = fn_outputs
        self._outputs: typing.List[mtf.Tensor] = [mtf.Tensor(self, x.shape, x.dtype, index=i)
                                                  for i, x in enumerate(fn_outputs)]
        self._forward_operations = forward_operations

    def lower(self, lowering):
        for fn_output, output in zip(self._fn_outputs, self._outputs):
            lowering.set_tensor_lowering(output, lowering.tensors[fn_output])

    def gradient(self, grad_ys: typing.List[mtf.Tensor],
                 params: typing.Optional[typing.List[mtf.Operation]] = None):
        dx, x_backwards, dv, v_backwards = grad_ys
        x: mtf.Tensor = self._x if x_backwards is None else x_backwards
        v: mtf.Tensor = self._v if v_backwards is None else v_backwards
        f_again_ops, mapping = self._graph.clone_operations(self._forward_operations, {self._x: x})
        # figure out what Tensors are downstream of xs
        downstream = set([x] + self._variables)
        for op in f_again_ops:
            if op.has_gradient and set(op.inputs) & downstream:
                downstream |= set(op.outputs)
        fx = mapping[self._fx]
        tensor_to_gradient = {mapping[self.outputs[0]]: dx, mapping[self.outputs[2]]: dv}
        if params is None:
            with tf1.variable_scope(fx.graph.captured_variable_scope):
                for op in f_again_ops[::-1]:
                    grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
                    if not op.has_gradient or not any(grad_outputs) or not set(op.inputs) & downstream:
                        continue
                    with tf1.variable_scope(op.name + "/revnet/gradients"):
                        for inp, grad in zip(op.inputs, op.gradient(grad_outputs)):
                            if inp not in downstream or grad is None:
                                continue
                            if inp in tensor_to_gradient:
                                tensor_to_gradient[inp] += grad
                            else:
                                tensor_to_gradient[inp] = grad
            yield dx + (1 - self.params.momentumnet_alpha) * tensor_to_gradient[x]
            yield x - v
            yield (dx + dv) * self.params.momentumnet_alpha
            yield (v - (1 - self.params.momentumnet_alpha) * fx) / self.params.momentumnet_alpha
            yield from (tensor_to_gradient.get(x) for x in self._variables)
            return
        tensor_to_gradient = {mapping[self.outputs[0]]: [0, 0, dx], mapping[self.outputs[2]]: [0, 0, dv]}
        yield params[1], x - v
        yield params[2], (dx + dv) * self.params.momentumnet_alpha
        yield params[3], (v - (1 - self.params.momentumnet_alpha) * fx) / self.params.momentumnet_alpha
        with tf1.variable_scope(fx.graph.captured_variable_scope):
            for op in f_again_ops[::-1]:
                grad_outputs = []
                for out in op.outputs:
                    grad = tensor_to_gradient.get(out)
                    if grad is None:
                        grad_outputs.append(None)
                        continue
                    grad_outputs.append(grad[2])
                    grad[0] += 1
                    if grad[0] == len(grad[2].operation.inputs):
                        del tensor_to_gradient[out]
                if not op.has_gradient or not any(grad_outputs) or not set(op.inputs) & downstream:
                    continue
                for inp, grad in zip(op.inputs, op.gradient(grad_outputs)):
                    if inp not in downstream or grad is None:
                        continue
                    if inp in tensor_to_gradient:
                        grad_list = tensor_to_gradient[inp]
                        grad_list[1] += 1
                        with tf1.variable_scope(op.name + "/revnet/gradients"):
                            grad_list[2] += grad
                    else:
                        tensor_to_gradient[inp] = grad_list = [0, 1, grad]
                    if len(inp.operation.outputs) != grad_list[1]:
                        continue
                    if inp not in self._variables:
                        continue
                    yield params[4 + self._variables.index(inp)], grad_list[2]
        yield params[2], dx + (1 - self.params.momentumnet_alpha) * tensor_to_gradient[x]
