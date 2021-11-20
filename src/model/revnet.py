import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .frontend import block_part_fn
from ..mtf_wrapper import add
from ..utils_core import random_name
from ..utils_mtf import gradient_iterator

tf1 = tf.compat.v1


class RevGradOp(mtf.Operation):
    """Operation to implement custom gradients.

    See comments on custom_gradient() below.
    """

    def __init__(self, params, block_config, x1, x1_backwards, x2, x2_backwards, index):
        graph: mtf.Graph = x1.graph
        prev_ops = len(graph.operations)
        y1 = add(x1, block_part_fn(params, block_config, x2, index))
        fn_outputs = [x2, x2_backwards, y1, x1_backwards]
        forward_operations = graph.operations[prev_ops:]
        new_outputs = set()
        new_inputs = set()
        for op in forward_operations:
            new_inputs.update(set(op.inputs))
            if not isinstance(op, mtf.Variable):
                new_outputs.update(set(op.outputs))
        explicit_inputs = [x1, x1_backwards, x2, x2_backwards]
        variables = [t for t in list(new_inputs - new_outputs - set(explicit_inputs)) if t.dtype.is_floating]
        super(RevGradOp, self).__init__(explicit_inputs + variables + fn_outputs, x1.mesh,
                                        random_name("custom_gradient"))
        # Make sure no one uses the internals of this function, since the gradients
        #  will probably not work correctly.
        for t in new_outputs - set(fn_outputs):
            t.usable = False

        self._graph: mtf.Graph = x1.graph
        self._x2: mtf.Tensor = x2
        self._y1: mtf.Tensor = y1
        self._variables: typing.List[mtf.Variable] = variables
        self._fn_outputs: typing.List[mtf.Tensor] = fn_outputs
        self._outputs: typing.List[mtf.Tensor] = [mtf.Tensor(self, x.shape, x.dtype, index=i)
                                                  for i, x in enumerate(fn_outputs)]
        self._forward_operations = forward_operations[:-1]

    def lower(self, lowering):
        for fn_output, output in zip(self._fn_outputs, self._outputs):
            lowering.set_tensor_lowering(output, lowering.tensors[fn_output])

    def gradient(self, grad_ys, params: typing.Optional[typing.List[mtf.Operation]] = None):
        dy2, dy2_backwards, dy1, dy1_backwards = grad_ys
        x2 = self._x2 if dy2_backwards is None else dy2_backwards
        f_again_ops, mapping = self._graph.clone_operations(self._forward_operations, {self._x2: x2})
        fx2 = mapping[self._forward_operations[-1].outputs[0]]
        # figure out what Tensors are downstream of xs
        downstream = set([x2] + self._variables)
        for op in f_again_ops:
            if op.has_gradient and set(op.inputs) & downstream:
                downstream |= set(op.outputs)
        tensor_to_gradient = {fx2: dy1}
        if params is None:
            yield dy1
            yield (self._y1 if dy1_backwards is None else dy1_backwards) - fx2
            with tf1.variable_scope(fx2.graph.captured_variable_scope):
                for op in f_again_ops[::-1]:
                    grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
                    if not op.has_gradient or not any(grad_outputs) or not set(op.inputs) & downstream:
                        continue
                    with tf1.variable_scope(op.name + "/revnet/gradients"):
                        for inp, grad in gradient_iterator(self.params, op, grad_outputs):
                            if inp not in downstream or grad is None:
                                continue
                            if inp in tensor_to_gradient:
                                tensor_to_gradient[inp] = add(tensor_to_gradient[inp], grad)
                            else:
                                tensor_to_gradient[inp] = grad
            yield add(dy2, tensor_to_gradient[x2])
            yield x2
            yield from (tensor_to_gradient.get(x) for x in self._variables)
            return
        tensor_to_gradient = {fx2: [0, 0, dy1]}
        yield params[0], dy1
        yield params[1], (self._y1 if dy1_backwards is None else dy1_backwards) - fx2
        yield params[3], x2
        with tf1.variable_scope(fx2.graph.captured_variable_scope):
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
                for inp, grad in gradient_iterator(self.params, op, grad_outputs):
                    if inp not in downstream or grad is None:
                        continue
                    if inp in tensor_to_gradient:
                        grad_list = tensor_to_gradient[inp]
                        grad_list[1] += 1
                        with tf1.variable_scope(op.name + "/revnet/gradients"):
                            grad_list[2] = add(grad_list[2], grad)
                    else:
                        tensor_to_gradient[inp] = grad_list = [0, 1, grad]
                    if len(inp.operation.outputs) != grad_list[1]:
                        continue
                    if inp not in self._variables:
                        continue
                    yield params[4 + self._variables.index(inp)], grad_list[2]
        yield params[2], add(dy2, tensor_to_gradient[x2][2])
