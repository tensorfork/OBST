"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""
from __future__ import absolute_import, division, print_function

import typing

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter
from .utils_mtf import weighted_add


def get_optimizer(loss: mtf.Tensor, params: ModelParameter
                  ) -> typing.Tuple[typing.Tuple[mtf.Tensor, typing.List[mtf.Assign],
                                                 typing.List[mtf.Tensor]], tf.Tensor]:
    """
    Creates optimizing and update/training operations.
    :param loss: Final scalar loss of the model
    :param params: ModelParameter instance
    :return: scalar learning rate, update operations, gradients
    """
    global_step = tf.train.get_or_create_global_step()
    dtype = params.variable_dtype.activation_dtype
    tf_learning_rate = tf.constant(value=params.learning_rate, shape=[], dtype=tf.float32)
    global_steps_float = tf.cast(global_step, tf.float32)

    if params.warmup_steps > 0:
        warmup_steps_float = tf.constant((params.warmup_steps * params.grad_accumulation), dtype=tf.float32)
        is_warmup = tf.cast(global_steps_float < warmup_steps_float, tf.float32)
        tf_learning_rate = (tf_learning_rate * (is_warmup * global_steps_float / warmup_steps_float + 1 - is_warmup))

    if params.learning_rate_decay_multi != 0 and params.learning_rate_decay_multi != 1:
        is_decay = tf.cast(tf.math.logical_and(global_steps_float > tf.constant(params.learning_rate_decay_start_step,
                                                                                tf.float32),
                                               tf_learning_rate > tf.constant(params.learning_rate_decay_min,
                                                                              tf.float32)), tf.float32)
        tf_learning_rate = tf_learning_rate * weighted_add(tf.constant(params.learning_rate_decay_multi,
                                                                       tf.float32) ** global_steps_float, 1, is_decay)

    learning_rate = mtf.import_fully_replicated(params.mesh, tf.cast(tf_learning_rate,
                                                                     dtype),
                                                [], "learning_rate")
    global_step = mtf.import_fully_replicated(params.mesh, tf.cast(tf.train.get_or_create_global_step(),
                                                                   params.variable_dtype.activation_dtype),
                                              [], "global_steps_float")
    step = mtf.cast(mtf.equal(mtf.mod(global_step, params.grad_accumulation), 0), dtype)
    mstep = 1 - step
    beta1 = 1 - step * mtf.import_fully_replicated(params.mesh, tf.constant(1 - 0.9, dtype, []),
                                                   mtf.Shape([]), name="beta1")
    beta2 = 1 - step * mtf.import_fully_replicated(params.mesh, tf.constant(1 - 0.95, dtype, []),
                                                   mtf.Shape([]), name="beta1")
    epsilon = 1e-5

    def variable(var, name, shape):
        return mtf.get_variable(var.mesh, f"{var.name}/{params.optimizer}/{name}", shape,
                                initializer=tf.zeros_initializer(), trainable=False, dtype=params.variable_dtype)

    update_ops = []
    operations = loss.graph.operations
    xs = [x.outputs[0] for x in params.mesh.graph.trainable_variables]
    tensor_to_var = dict(zip(xs, params.mesh.graph.trainable_variables))
    loss_grad = mtf.Constant(loss.mesh, 1.0, loss.shape, loss.dtype).outputs[0]
    downstream = set(xs)
    for op in operations:
        if op.has_gradient and (set(op.inputs) & downstream):
            downstream |= set(op.outputs)
    tensor_to_gradient: typing.Dict[mtf.Tensor, typing.List[int, int, mtf.Tensor]] = {loss: [0, 0, loss_grad]}
    with tf.variable_scope(loss.graph.captured_variable_scope):
        for op in operations[::-1]:
            grad_outputs = []
            for out in op.outputs:
                grad = tensor_to_gradient.get(out)
                if grad is not None:
                    grad_outputs.append(grad[2])
                    grad[0] += 1
                else:
                    grad_outputs.append(None)
                if grad is not None and grad[0] == len(grad[2].operation.inputs):
                    del tensor_to_gradient[out]
            if not op.has_gradient or not any(grad_outputs) or not (set(op.inputs) & downstream):
                continue
            with tf.variable_scope(op.name + "/gradients"):
                for inp, grad in zip(op.inputs, op.gradient(grad_outputs)):

                    valid_grad = inp in downstream and grad is not None
                    if valid_grad and inp in tensor_to_gradient:
                        grad_list = tensor_to_gradient[inp]
                        grad_list[1] += 1
                        grad_list[2] += grad

                    elif valid_grad:
                        grad_list = [0, 1, grad]
                        tensor_to_gradient[inp] = grad_list

                    if valid_grad and len(inp.operation.outputs) == grad_list[1] and inp in tensor_to_var:
                        grad = grad_list[2]
                        var: mtf.Variable = tensor_to_var[inp]
                        if params.grad_accumulation > 1:
                            grad_buffer = variable(var, "grad_accumulation", var.shape)
                            update_ops.append(mtf.assign(grad_buffer, grad + grad_buffer * mstep))
                            grad = grad_buffer * step / params.grad_accumulation
                        if params.gradient_clip > 0:
                            grd_norm = mtf.sqrt(mtf.reduce_sum(mtf.square(grad)) + 1e-5)
                            wgt_norm = mtf.sqrt(mtf.reduce_sum(mtf.square(var.value)) + 1e-3)
                            grad = weighted_add(grd_norm / wgt_norm * params.gradient_clip * grad, grad,
                                                mtf.cast(mtf.greater(wgt_norm / grd_norm, params.gradient_clip), dtype))
                        if var.shape.ndims <= 1 or params.optimizer == 'adam':
                            exp_avg_p1_ptr = variable(var, 'exp_avg_p1', var.shape)
                            exp_avg_p2_ptr = variable(var, 'exp_avg_p2', var.shape)

                            exp_avg_p1 = weighted_add(exp_avg_p1_ptr, grad, beta1)
                            exp_avg_p2 = weighted_add(exp_avg_p2_ptr, mtf.square(grad), beta2)

                            weight_update = exp_avg_p1 * mtf.rsqrt(exp_avg_p2 + epsilon)
                            update_ops.extend([mtf.assign(exp_avg_p1_ptr, exp_avg_p1),
                                               mtf.assign(exp_avg_p2_ptr, exp_avg_p2)])
                        elif params.optimizer == 'sgd':
                            weight_update = grad
                        elif params.optimizer == 'novograd':
                            exp_avg_p1 = exp_avg_p1_ptr = variable(var, "exp_avg_p1", var.shape)
                            exp_avg_p2 = exp_avg_p2_ptr = variable(var, "exp_avg_p2", [])

                            exp_avg_p2 = weighted_add(exp_avg_p2, mtf.reduce_sum(mtf.square(grad)), beta2)
                            weight_update = beta1 * exp_avg_p1 + grad * mtf.rsqrt(exp_avg_p2 + epsilon)
                            update_ops.extend([mtf.assign(exp_avg_p1_ptr, beta1 * exp_avg_p1_ptr +
                                                          grad * mtf.rsqrt(exp_avg_p2 + epsilon)),
                                               mtf.assign(exp_avg_p2_ptr, exp_avg_p2)])
                        elif params.optimizer == 'sm3':
                            update = variable(var, "dim0", [var.shape.dims[0]])
                            buffer = [update]

                            for i in range(1, var.shape.ndims):
                                buffer.append(variable(var, f"dim{i}", [var.shape.dims[i]]))
                                update = mtf.minimum(update, buffer[-1])

                            update += mtf.square(grad)

                            weight_update = grad * mtf.rsqrt(update + epsilon)
                            update_ops.extend([mtf.assign(buf_ptr, mtf.reduce_max(update, output_shape=[dim]))
                                               for buf_ptr, dim in zip(buffer, update.shape.dims)])
                        weight_update *= learning_rate
                        if params.weight_decay > 0:
                            weight_update += params.weight_decay * var.value
                        if var.shape.size > 1:
                            weight_update += mtf.reduce_mean(var.value)
                        if params.grad_accumulation > 1:
                            weight_update *= step
                        update_ops.append(mtf.assign_sub(var, weight_update))

    return params.mesh.graph.trainable_variables[0].graph.combine_assignments(update_ops), tf_learning_rate
