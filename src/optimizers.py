"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf2
from tensorflow.python.ops.init_ops import Initializer

from src.model.revnet import RevGradOp
from .dataclass import ModelParameter
from .mtf_wrapper import (add_n, cast, constant_float, constant_scalar, einsum, equal, greater, greater_equal, minimum,
                          mod, reduce_max, reduce_mean, reduce_sum, rsqrt, sqrt, square, assign, assign_sub,
                          one_hot as mtf_one_hot, logical_and, add, multiply, import_fully_replicated,
                          reshape, scoped, assign_add, maximum, identity, zeros_like)
from .utils_mtf import SHAPE, feature_dims_used, to_fp32, weighted_add, get_variable, get_fan_in

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def import_float(imported):
    return tf.constant(imported, dtype=tf.float32, shape=[])


def get_var(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer = zeros):
    return get_variable(params, name, shape, initializer, False, params.optimizer_dtype)


def variable(params: ModelParameter, base: mtf.Variable, name: str, shape: SHAPE):
    return get_variable(params, f"{base.name}/{params.optimizer}/{name}", shape, zeros, False, params.optimizer_dtype)


class OptimizerCtx:
    def __init__(self, op: mtf.Operation, grad_outputs: typing.List[mtf.Tensor], downstream: typing.Set[mtf.Operation],
                 tensor_to_gradient: dict, tensor_to_var: dict, params: ModelParameter, loss_idx: int, update_ops: list,
                 debug_gradients_dict: dict, loss_list: list, first_grad: dict,
                 loss_1__loss_1: typing.Optional[mtf.Tensor], loss_1__loss_2: typing.Optional[mtf.Tensor],
                 loss_2__loss_2: typing.Optional[mtf.Tensor], mstep: mtf.Tensor, step: mtf.Tensor,
                 dtype: mtf.VariableDType, beta1: mtf.Tensor, beta2: mtf.Tensor, epsilon: mtf.Tensor,
                 learning_rate: mtf.Tensor):
        self.op = op
        self.grad_outputs = grad_outputs
        self.downstream = downstream
        self.tensor_to_gradient = tensor_to_gradient
        self.tensor_to_var = tensor_to_var
        self.params = params
        self.loss_idx = loss_idx
        self.update_ops = update_ops
        self.debug_gradients_dict = debug_gradients_dict
        self.loss_list = loss_list
        self.first_grad = first_grad
        self.loss_1__loss_1 = loss_1__loss_1
        self.loss_1__loss_2 = loss_1__loss_2
        self.loss_2__loss_2 = loss_2__loss_2
        self.mstep = mstep
        self.step = step
        self.dtype = dtype
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.args = [op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params, loss_idx, update_ops,
                     debug_gradients_dict, loss_list, first_grad, loss_1__loss_1, loss_1__loss_2, loss_2__loss_2, mstep,
                     step, dtype, beta1, beta2, epsilon, learning_rate]

        self.var: typing.Optional[mtf.Variable] = None
        self.grad_buffer: typing.Optional[mtf.Variable] = None
        self.grad: typing.Optional[mtf.Tensor] = None

    def __call__(self, var: mtf.Variable, grad: mtf.Tensor):
        self.var, self.grad = var, grad
        return self


def gradients(ctx: OptimizerCtx):
    op = ctx.op
    grad_outputs = ctx.grad_outputs
    downstream = ctx.downstream
    tensor_to_gradient = ctx.tensor_to_gradient
    tensor_to_var = ctx.tensor_to_var
    params = ctx.params
    first_grad = ctx.first_grad
    loss_list = ctx.loss_list
    loss_idx = ctx.loss_idx
    update_ops = ctx.update_ops
    debug_gradients_dict = ctx.debug_gradients_dict

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
            flat_grad = variable(params, var, f"loss_{loss_idx}", flat_shape)
            update_ops.append(assign(flat_grad, reshape(grad, new_shape=flat_shape)))
            debug_gradients_dict[f"loss_{loss_idx}/{var.name}"] = flat_grad

        if len(loss_list) > 1:
            if params.multi_loss_strategy == "pcgrad":
                if 'body' in op.name:

                    if loss_idx < len(loss_list) - 1:
                        first_grad[op.name] = grad
                        continue

                    else:

                        all_grads = [grad, first_grad[op.name]]
                        g_square = [1e-6 + einsum([g, g], output_shape=[]) for g in all_grads[1:]]

                        for i in range(len(all_grads)):
                            grad = all_grads.pop(0)
                            for g, sq in zip(all_grads, g_square):
                                grad -= g * (minimum(einsum([grad, g], output_shape=[]), 0) / sq)

                            all_grads.append(grad)
                            g_square.append(einsum([g, g], output_shape=[]))

                        grad = add_n(all_grads)

                elif params.multi_loss_strategy == "mgda":
                    if 'body' in op.name:
                        if loss_idx < 2:
                            if loss_idx == 0:
                                first_grad[op.name] = grad
                                continue

                            else:

                                ctx.loss_1__loss_1 += einsum([first_grad[op.name], first_grad[op.name]],
                                                             [params.head_dim])
                                ctx.loss_1__loss_2 += einsum([first_grad[op.name], grad], [params.head_dim])
                                ctx.loss_2__loss_2 += einsum([grad, grad], [params.head_dim])

                                del first_grad[op.name]
                                continue

                    elif loss_idx == 2:  # not in body and optimize body params.
                        continue
        ctx(var, grad)
        yield None


def gradient_accumulation(ctx: OptimizerCtx):
    ctx.update_ops.append(assign_add(ctx.grad_buffer, ctx.grad + identity(ctx.grad_buffer)))


def adam(ctx: OptimizerCtx) -> mtf.Tensor:
    exp_avg_p2_ptr = variable(ctx.params, ctx.var, 'exp_avg_p2', ctx.var.shape)
    exp_avg_p2 = weighted_add(exp_avg_p2_ptr, square(ctx.grad), ctx.beta2)
    ctx.update_ops.append(assign(exp_avg_p2_ptr, exp_avg_p2))
    if ctx.params.opt_beta1:
        exp_avg_p1_ptr = variable(ctx.params, ctx.var, 'exp_avg_p1', ctx.var.shape)
        grad = weighted_add(exp_avg_p1_ptr, ctx.grad, ctx.beta1)
        ctx.update_ops.append(assign(exp_avg_p1_ptr, grad))
    return ctx.grad * rsqrt(exp_avg_p2 + ctx.epsilon)


def novograd(ctx: OptimizerCtx) -> mtf.Tensor:
    exp_avg_p1 = exp_avg_p1_ptr = variable(ctx.params, ctx.var, "exp_avg_p1", ctx.var.shape)
    exp_avg_p2 = exp_avg_p2_ptr = variable(ctx.params, ctx.var, "exp_avg_p2", [])

    exp_avg_p2 = weighted_add(exp_avg_p2, reduce_sum(square(ctx.grad)), ctx.beta2)
    ctx.update_ops.extend([assign(exp_avg_p1_ptr, ctx.beta1 * exp_avg_p1_ptr +
                                  ctx.grad * rsqrt(exp_avg_p2 + ctx.epsilon)),
                           assign(exp_avg_p2_ptr, exp_avg_p2)])
    return ctx.beta1 * exp_avg_p1 + ctx.grad * rsqrt(exp_avg_p2 + ctx.epsilon)


def sm3(ctx: OptimizerCtx) -> mtf.Tensor:
    weight_update = variable(ctx.params, ctx.var, "dim0", [ctx.var.shape.dims[0]])
    buffer = [weight_update]

    for i in range(1, ctx.var.shape.ndims):
        buffer.append(variable(ctx.params, ctx.var, f"dim{i}", [ctx.var.shape.dims[i]]))
        weight_update = minimum(weight_update, buffer[-1])

    weight_update += square(ctx.grad)

    ctx.update_ops.extend([assign(buf_ptr, reduce_max(weight_update, output_shape=[dim]))
                           for buf_ptr, dim in zip(buffer, weight_update.shape.dims)])
    return ctx.grad * rsqrt(weight_update + ctx.epsilon)


def return_grad(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad


def adaptive_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    grd_norm = sqrt(einsum([ctx.grad, ctx.grad], reduced_dims=get_fan_in(ctx.params, ctx.var)) + 1e-5)
    wgt_norm = sqrt(einsum([ctx.var.value, ctx.var.value], reduced_dims=get_fan_in(ctx.params, ctx.var)) + 1e-3)
    return weighted_add(grd_norm / wgt_norm * gradient_clip * ctx.grad, ctx.grad,
                        cast(greater(wgt_norm / grd_norm, gradient_clip), ctx.dtype.activation_dtype))


def norm_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return einsum([minimum(rsqrt(einsum([ctx.grad, ctx.grad], []) + 1e-6), 1 / gradient_clip),
                   ctx.grad, constant_scalar(ctx.params, gradient_clip)], ctx.grad.shape)


def value_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return maximum(minimum(ctx.grad, gradient_clip), -gradient_clip)


def gradient_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad - reduce_mean(ctx.grad)


def weight_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad + reduce_mean(ctx.var.value)


def multiply_learning_rate(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad * ctx.learning_rate


OPTIMIZERS = {"adam": adam,
              "sm3": sm3,
              "novograd": novograd,
              "sgd": return_grad,
              "adaptive_clip": adaptive_gradient_clipping,
              "norm_clip": norm_gradient_clipping,
              "value_clip": value_gradient_clipping,
              "gradient_centralisation": gradient_centralisation,
              "weight_centralisation": weight_centralisation,
              "learning_rate": multiply_learning_rate
              }


def update(ctx: OptimizerCtx):
    params = ctx.params
    update_ops = ctx.update_ops
    learning_rate = ctx.learning_rate

    var = ctx.var
    if ctx.grad_buffer is not None:
        ctx.grad = identity(ctx.grad_buffer.value)
        ctx.update_ops.append(assign(ctx.grad_buffer, zeros_like(ctx.grad)))

    for opt in params.optimizer.split('-'):
        opt, *args = opt.split(':')
        ctx.grad = scoped(opt, OPTIMIZERS[opt], ctx, *args)

    if 'rezero' in var.name:
        ctx.grad *= params.rezero_lr_multiplier

    large_tensor = feature_dims_used(params, var) and len(var.shape.dims) > len(params.feature_dims)
    large_tensor |= not feature_dims_used(params, var) and len(var.shape.dims) >= 2
    large_tensor &= var.shape.size > 1
    large_tensor &= params.vocab_dim not in var.shape

    if large_tensor and params.weight_decay > 0:
        ctx.grad += params.weight_decay * var.value * learning_rate

    if not large_tensor or not params.weight_standardisation:
        update_ops.append(assign_sub(var, ctx.grad))
        return

    val: mtf.Tensor = var.value - ctx.grad
    fan_in_size = np.prod([d.size for d in get_fan_in(params, var)])
    size = np.prod([d.size for d in var.shape.dims])
    max_fan = max(fan_in_size, size // fan_in_size)
    var = ((1 - 1 / max_fan) / size ** 2 + 1 / max_fan - 2 / size + 1 / max_fan)
    if params.scale_by_depth:
        var *= params.n_blocks
    std = rsqrt(1e-6 + reduce_sum(square(val * val.size ** -0.5), output_shape=[]))
    std *= var ** 0.5
    update_ops.append(assign(var, val * std))


def import_mtf(params: ModelParameter, imported: typing.Union[tf.Tensor, float], name: str):
    return import_fully_replicated(params, tf.cast(imported, params.variable_dtype.activation_dtype), [], name)


def get_optimizer(loss_list: typing.List[mtf.Tensor], params: ModelParameter, manual_step: tf.Tensor, fn: str
                  ) -> typing.Tuple[typing.Tuple[mtf.Tensor, typing.List[mtf.Assign], typing.List[mtf.Tensor]],
                                    tf.Tensor, typing.Dict]:
    """
    Creates optimizing and update/training operations.
    :param loss_list: Final scalar loss of the model
    :param params: ModelParameter instance
    :param manual_step: manually incremented global_step variable to account for grad accumulation
    :param fn: whether to "accumulate" gradients or "update" parameters.
    :return: scalar learning rate, update operations, gradients

    there is no check for "update". you can just call it "oijhiojio" and it'll still work. just make sure it's not
    called "accumulate".
    """

    global_step = tf.train.get_or_create_global_step()
    dtype = params.variable_dtype.activation_dtype
    tf_learning_rate = tf.constant(value=params.learning_rate, shape=[], dtype=tf.float32)
    global_steps_float = tf.cast(global_step, tf.float32)
    global_step_mtf = import_fully_replicated(params, global_step, [], "mtf_learning_rate")
    update_ops = []

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

    if params.reduce_lr_on_plateau_timespan:
        base = "reduce_lr_on_plateau/"
        loss_sum = add_n(loss_list)
        window_dim = mtf.Dimension("loss_window", params.reduce_lr_on_plateau_timespan)

        divisor_ptr = get_var(params, f"{base}lr_divisor", [], tf.ones_initializer())
        loss_window_ptr = get_var(params, f"{base}loss_window", [window_dim])
        loss_ema_ptr = get_var(params, f"{base}loss_ema", [])
        last_reduce = get_var(params, f"{base}last_reduce", [])

        one_hot = mtf_one_hot(mod(global_step_mtf, params.reduce_lr_on_plateau_timespan), window_dim)
        sub = (loss_sum - loss_window_ptr) * one_hot
        loss_window = loss_window_ptr - sub
        window_mean = reduce_mean(loss_window, output_shape=[])
        loss_ema = loss_ema_ptr * (2 / params.reduce_lr_on_plateau_timespan)
        loss_ema += loss_sum * (1 - 2 / params.reduce_lr_on_plateau_timespan)
        reduce = cast(logical_and(greater(global_step_mtf, last_reduce + params.reduce_lr_on_plateau_timespan),
                                  greater(window_mean, loss_ema)),
                      params.variable_dtype.activation_dtype)
        reduce = reduce * (params.reduce_lr_on_plateau_reduction - 1) + 1
        divisor = divisor_ptr * reduce
        tf_learning_rate /= divisor

        update_ops.append(assign(divisor_ptr, divisor))
        update_ops.append(assign_sub(loss_window_ptr, sub))
        update_ops.append(assign(loss_ema_ptr, loss_ema))
        update_ops.append(assign(last_reduce, weighted_add(last_reduce, global_step_mtf, reduce)))

    learning_rate = import_mtf(params, tf_learning_rate, "learning_rate")
    step = cast(equal(mod(tf.cast(manual_step + 1, dtype),
                          import_mtf(params, params.grad_accumulation * 1., "grad_accum")),
                      import_mtf(params, 0., "zero")), dtype)
    mstep = 1 - step
    beta1 = 1 - step * import_mtf(params, 1 - params.opt_beta1, "beta1") if params.opt_beta1 else None
    beta2 = 1 - step * import_mtf(params, 1 - params.opt_beta2, "beta2")
    epsilon = params.opt_epsilon

    debug_gradients_dict = {}
    first_grad = {}
    loss_1__loss_1 = loss_1__loss_2 = loss_2__loss_2 = 0
    if params.multi_loss_strategy == "mgda":
        loss_1__loss_1 = constant_float(params, 0, shape=[params.head_dim])
        loss_1__loss_2 = constant_float(params, 0, shape=[params.head_dim])
        loss_2__loss_2 = constant_float(params, 0, shape=[params.head_dim])

    for loss_idx, loss in enumerate(loss_list):

        if params.multi_loss_strategy == "mgda" and loss_idx == 2:
            v1v1 = reduce_sum(loss_1__loss_1, output_shape=[])
            v1v2 = reduce_sum(loss_1__loss_2, output_shape=[])
            v2v2 = reduce_sum(loss_2__loss_2, output_shape=[])
            min_gamma = 0.001
            gamma = constant_float(params, value=(1 - min_gamma), shape=[]) * to_fp32(greater_equal(v1v2, v1v1))
            gamma += constant_float(params, value=min_gamma, shape=[]) * \
                     to_fp32(greater_equal(v1v2, v2v2)) * to_fp32(equal(gamma, 0))
            gamma += (-1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))) * to_fp32(equal(gamma, 0))

            loss = add(multiply(loss_list[0], gamma),
                       multiply(loss_list[1], (1 - gamma)))

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
                ctx = OptimizerCtx(op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params,
                                   loss_idx, update_ops, debug_gradients_dict, loss_list, first_grad, loss_1__loss_1,
                                   loss_1__loss_2, loss_2__loss_2, mstep, step, dtype, beta1, beta2, epsilon,
                                   learning_rate)
                for _ in gradients(ctx):
                    full_name = f'{tf.get_variable_scope().name}/f"{ctx.var.name}/{params.optimizer}/grad_accumulation'
                    if fn == "accumulate" or full_name in params.mesh.graph.name_to_variable:
                        ctx.grad_buffer = variable(params, ctx.var, "grad_accumulation", ctx.var.shape)
                    scoped(fn, gradient_accumulation if "accumulate" else update, ctx)
    return params.mesh.graph.trainable_variables[0].graph.combine_assignments(update_ops), \
           tf_learning_rate, debug_gradients_dict
