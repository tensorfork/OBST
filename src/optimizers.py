"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .dataclass import ModelParameter
from .model import RevGradOp
from .utils_mtf import (add_n, anonymize, anonymize_dim, cast, constant_scalar, einsum, equal, greater, minimum, mod,
                        reduce_max, reduce_mean, reduce_sum, rsqrt, sqrt, square, weighted_add, feature_dims_used)


def import_float(imported):
    return tf.constant(imported, dtype=tf.float32, shape=[])


def sum_dim(inp: mtf.Tensor, dims: typing.List[mtf.Dimension]):
    return einsum([inp, anonymize(dims, dims)], output_shape=mtf.Shape(dims) + [anonymize_dim(d) for d in dims])


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

    global_step = tf.train.get_or_create_global_step()
    dtype = params.variable_dtype.activation_dtype
    tf_learning_rate = tf.constant(value=params.learning_rate, shape=[], dtype=tf.float32)
    global_steps_float = tf.cast(global_step, tf.float32)

    def import_mtf(imported, name):
        return mtf.import_fully_replicated(params.mesh, tf.cast(imported, dtype), [], name)

    if params.warmup_steps > 0:
        warmup_steps_float = import_float(params.warmup_steps)
        is_warmup = tf.cast(global_steps_float < warmup_steps_float, tf.float32)
        tf_learning_rate = tf_learning_rate * weighted_add(global_steps_float / warmup_steps_float, 1, is_warmup)

    if params.learning_rate_decay_multi != 0 and params.learning_rate_decay_multi != 1:
        start_step = import_float(params.learning_rate_decay_start_step * 1.)
        tf_learning_rate = tf.maximum(tf_learning_rate *
                                      import_float(params.learning_rate_decay_multi * 1.) **
                                      (tf.maximum(global_steps_float - start_step, import_float(0.))),
                                      import_float(params.learning_rate_decay_min * 1.))

    learning_rate = import_mtf(tf_learning_rate, "learning_rate")
    step = cast(equal(mod(tf.cast(manual_step, dtype),
                          import_mtf(params.grad_accumulation * 1., "grad_accum")),
                      import_mtf(0., "zero")), dtype)
    mstep = 1 - step
    beta1 = 1 - step * import_mtf(1 - 0.9, "beta1")
    beta2 = 1 - step * import_mtf(1 - 0.95, "beta2")
    epsilon = 1e-5

    def variable(var, name, shape):
        return mtf.get_variable(var.mesh, f"{var.name}/{params.optimizer}/{name}", shape,
                                initializer=tf.zeros_initializer(), trainable=False, dtype=params.variable_dtype)

    debug_gradients_dict = {}

    for loss_idx, loss in enumerate(loss_list):

        update_ops = []
        operations = loss.graph.operations
        xs = [x.outputs[0] for x in params.mesh.graph.trainable_variables]
        tensor_to_var = dict(zip(xs, params.mesh.graph.trainable_variables))
        loss_grad = constant_scalar(params, 1.0)
        downstream = set(xs)

        for op in operations:
            if op.has_gradient and (set(op.inputs) & downstream):
                downstream |= set(op.outputs)

        tensor_to_gradient: typing.Dict[mtf.Tensor, typing.List[int, int, mtf.Tensor]] = {loss: [0, 0, loss_grad]}

        with tf.variable_scope(loss.graph.captured_variable_scope):
            for op in operations[::-1]:
                grad_outputs = []
                for out in op.outputs:
                    if out not in tensor_to_gradient:
                        grad_outputs.append(None)
                        continue

                    grad_list: typing.Tuple[int, int, mtf.Tensor] = tensor_to_gradient[out]
                    grad_outputs.append(grad_list[2])
                    grad_list[0] += 1

                    if grad_list[0] == len(grad_list[2].operation.inputs):
                        del tensor_to_gradient[out]

                if not op.has_gradient or not any(grad_outputs) or not (set(op.inputs) & downstream):
                    continue

                with tf.variable_scope(op.name + "/optimizer/gradients"):
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
                            grad_list[2] += grad
                        else:
                            tensor_to_gradient[inp] = grad_list = [0, 1, grad]

                        if len(inp.operation.outputs) != grad_list[1] or inp not in tensor_to_var:
                            continue

                        grad: mtf.Tensor = grad_list[2]
                        var: mtf.Variable = tensor_to_var[inp]

                        if params.debug_gradients:
                            flat_shape = mtf.Shape([mtf.Dimension('flat_dim', var.size)])
                            flat_grad = variable(var, f"loss_{loss_idx}", flat_shape)
                            update_ops.append(mtf.assign(flat_grad, mtf.reshape(grad, new_shape=flat_shape)))
                            debug_gradients_dict[f"loss_{loss_idx}/{var.name}"] = flat_grad

                        if params.use_PCGrad and len(loss_list) > 1:
                            if 'body' in op.name:
                                other_grads = [variable(var, f"grad_{i}", var.shape) for i in range(len(loss_list) - 1)]
                                if loss_idx < len(loss_list) - 1:
                                    update_ops.append(mtf.assign(other_grads[loss_idx], grad))
                                    continue
                                other_grads.insert(0, grad)
                                g_square = [1e-6 + einsum([g, g], output_shape=[]) for g in other_grads[1:]]
                                for i in range(len(other_grads)):
                                    grad = other_grads.pop(0)
                                    for g, sq in zip(other_grads, g_square):
                                        grad -= g * (minimum(einsum([grad, g], output_shape=[]), 0) / sq)
                                    other_grads.append(grad)
                                    g_square.append(einsum([g, g], output_shape=[]))
                                grad = add_n(other_grads)

                        if params.grad_accumulation > 1:
                            grad_buffer = variable(var, "grad_accumulation", var.shape)
                            update_ops.append(mtf.assign(grad_buffer, grad + grad_buffer * mstep))
                            grad = grad_buffer * step / params.grad_accumulation

                        if params.gradient_clip > 0:
                            grd_norm = sqrt(reduce_sum(square(grad)) + 1e-5)
                            wgt_norm = sqrt(reduce_sum(square(var.value)) + 1e-3)
                            grad = weighted_add(grd_norm / wgt_norm * params.gradient_clip * grad, grad,
                                                cast(greater(wgt_norm / grd_norm, params.gradient_clip), dtype))

                        if var.shape.ndims <= 1 or params.optimizer == 'adam':
                            exp_avg_p1_ptr = variable(var, 'exp_avg_p1', var.shape)
                            exp_avg_p2_ptr = variable(var, 'exp_avg_p2', var.shape)

                            exp_avg_p1 = weighted_add(exp_avg_p1_ptr, grad, beta1)
                            exp_avg_p2 = weighted_add(exp_avg_p2_ptr, square(grad), beta2)

                            weight_update = exp_avg_p1 * rsqrt(exp_avg_p2 + epsilon)
                            update_ops.extend([mtf.assign(exp_avg_p1_ptr, exp_avg_p1),
                                               mtf.assign(exp_avg_p2_ptr, exp_avg_p2)])

                        elif params.optimizer == 'shampoo':
                            shape = grad.shape
                            buffers = []

                            if all(f in grad.shape.dims for f in params.feature_dims):
                                shape = shape - params.feature_dims
                                buffers.append(sum_dim(grad, params.feature_dims))
                            buffers.extend([sum_dim(grad, [d]) for d in shape.dims])

                        elif params.optimizer == 'sgd':
                            weight_update = grad

                        elif params.optimizer == 'novograd':
                            exp_avg_p1 = exp_avg_p1_ptr = variable(var, "exp_avg_p1", var.shape)
                            exp_avg_p2 = exp_avg_p2_ptr = variable(var, "exp_avg_p2", [])

                            exp_avg_p2 = weighted_add(exp_avg_p2, reduce_sum(square(grad)), beta2)
                            weight_update = beta1 * exp_avg_p1 + grad * rsqrt(exp_avg_p2 + epsilon)
                            update_ops.extend([mtf.assign(exp_avg_p1_ptr, beta1 * exp_avg_p1_ptr +
                                                          grad * rsqrt(exp_avg_p2 + epsilon)),
                                               mtf.assign(exp_avg_p2_ptr, exp_avg_p2)])

                        elif params.optimizer == 'sm3':
                            update = variable(var, "dim0", [var.shape.dims[0]])
                            buffer = [update]

                            for i in range(1, var.shape.ndims):
                                buffer.append(variable(var, f"dim{i}", [var.shape.dims[i]]))
                                update = minimum(update, buffer[-1])

                            update += square(grad)

                            weight_update = grad * rsqrt(update + epsilon)
                            update_ops.extend([mtf.assign(buf_ptr, reduce_max(update, output_shape=[dim]))
                                               for buf_ptr, dim in zip(buffer, update.shape.dims)])

                        if params.weight_decay > 0:
                            weight_update += params.weight_decay * var.value
                        weight_update *= learning_rate
                        features_used = feature_dims_used(params, var)
                        large_tensor = features_used and len(var.shape.dims) > len(params.feature_dims)
                        large_tensor |= not features_used and len(var.shape.dims) >= 2
                        large_tensor &= var.shape.size > 1
                        if params.weight_centralisation and large_tensor:
                            weight_update += reduce_mean(var.value)
                        if params.grad_accumulation > 1:
                            weight_update *= step
                        if params.weight_standardisation and large_tensor:
                            val: mtf.Tensor = var.value - weight_update
                            std = rsqrt(1e-6 + reduce_sum(square(val / (val.size ** 0.5)), output_shape=[]))

                            shape = [d.size for d in var.shape.dims]
                            if features_used and var.shape.dims.index(params.key_dim) == var.shape.ndims - 1:
                                fan_in = np.prod(shape[:-2])
                            elif features_used:
                                fan_in = np.prod(shape[:2])
                            else:
                                fan_in = shape[0]
                            size = np.prod(shape)
                            # ((1 - 1 / max_fan) / size ** 2 + 1 / max_fan - 2 / size + 1 / min_fan / size) ** 0.5
                            std *= ((fan_in - 2) / size / params.n_blocks) ** 0.5  # 0.01% error
                            update_ops.append(mtf.assign(var, val * std))
                        else:
                            update_ops.append(mtf.assign_sub(var, weight_update))

    return params.mesh.graph.trainable_variables[0].graph.combine_assignments(update_ops), \
           tf_learning_rate, debug_gradients_dict
