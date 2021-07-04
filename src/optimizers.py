"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""

import typing

import mesh_tensorflow as mtf
import tensorflow as tf2

from src.model.revnet import RevGradOp
from .dataclass import ModelParameter
from .mtf_wrapper import (constant_scalar, einsum, rsqrt, square, assign, assign_sub, add, import_fully_replicated,
                          scoped)
from .utils_mtf import SHAPE, weighted_add, get_variable

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def variable(params: ModelParameter, base: mtf.Variable, name: str, shape: SHAPE):
    return get_variable(params, f"{base.name}/{params.optimizer}/{name}", shape, zeros, False, params.optimizer_dtype)


def update(op: mtf.Operation, grad_outputs: typing.List[mtf.Tensor], downstream: typing.Set[mtf.Operation],
           tensor_to_gradient: dict, tensor_to_var: dict, params: ModelParameter, update_ops: list, beta1: mtf.Tensor,
           beta2: mtf.Tensor,
           learning_rate: mtf.Tensor):
    if isinstance(op, RevGradOp):
        itr = op.gradient(grad_outputs, params=op.inputs)
    else:
        itr = zip(op.inputs, op.gradient(grad_outputs))
    for inp, grad in itr:
        if inp not in downstream or grad is None:
            continue

        if inp in tensor_to_gradient:
            grad_list = tensor_to_gradient[inp]
            grad_list[1] += 1
            grad_list[2] = add(grad_list[2], grad)
        else:
            tensor_to_gradient[inp] = grad_list = [0, 1, grad]

        if len(inp.operation.outputs) != grad_list[1] or inp not in tensor_to_var:
            continue

        grad: mtf.Tensor = grad_list[2]
        var: mtf.Variable = tensor_to_var[inp]

        exp_avg_p2_ptr = variable(params, var, 'exp_avg_p2', var.shape)
        exp_avg_p1_ptr = variable(params, var, 'exp_avg_p1', var.shape)

        exp_avg_p2 = weighted_add(exp_avg_p2_ptr, square(grad), beta2)
        grad = weighted_add(exp_avg_p1_ptr, grad, beta1)

        weight_update = einsum([grad, rsqrt(add(exp_avg_p2, params.opt_epsilon)), learning_rate],
                               output_shape=grad.shape)

        update_ops.append(assign(exp_avg_p2_ptr, exp_avg_p2))
        update_ops.append(assign(exp_avg_p1_ptr, grad))
        update_ops.append(assign_sub(var, weight_update))


def get_optimizer(loss_list: typing.List[mtf.Tensor], params: ModelParameter, manual_step: tf.Tensor,
                  ) -> typing.Tuple[typing.Tuple[mtf.Tensor, typing.List[mtf.Assign], typing.List[mtf.Tensor]],
                                    tf.Tensor, typing.Dict]:
    """
    Creates optimizing and update/training operations.
    :param loss_list: Final scalar loss of the model
    :param params: ModelParameter instance
    :param manual_step: manually incremented global_step variable to account for grad accumulation
    :return: scalar learning rate, update operations, gradients
    """

    update_ops = []

    def import_mtf(imported, name):
        return import_fully_replicated(params, tf.cast(imported, params.variable_dtype.activation_dtype), [], name)

    learning_rate = import_mtf(params.learning_rate, "learning_rate")
    beta1 = import_mtf(params.opt_beta1, "beta1")
    beta2 = import_mtf(params.opt_beta2, "beta2")

    loss = loss_list[0]
    operations = loss.graph.operations
    xs = [x.outputs[0] for x in params.mesh.graph.trainable_variables]
    tensor_to_var = dict(zip(xs, params.mesh.graph.trainable_variables))
    downstream = set(xs)

    for op in operations:
        if op.has_gradient and (set(op.inputs) & downstream):
            downstream |= set(op.outputs)

    tensor_to_gradient = {loss: [0, 0, constant_scalar(params, 1.0)]}

    with tf.variable_scope(loss.graph.captured_variable_scope):
        for op in operations[::-1]:
            grad_outputs = []
            for out in op.outputs:
                if out not in tensor_to_gradient:
                    grad_outputs.append(None)
                    continue

                grad_list = tensor_to_gradient[out]
                grad_outputs.append(grad_list[2])
                grad_list[0] += 1

                if grad_list[0] == len(grad_list[2].operation.inputs):
                    del tensor_to_gradient[out]

            if not op.has_gradient or not any(grad_outputs) or not (set(op.inputs) & downstream):
                continue
            scoped("update", update, op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params,
                   update_ops, beta1, beta2, learning_rate)
    return params.mesh.graph.trainable_variables[0].graph.combine_assignments(update_ops), \
           params.learning_rate, {}
