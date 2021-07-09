"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf2

from .backend import import_mtf, import_float, get_var, variable
from .context import OptimizerCtx
from .gradients import gradients
from .learning_rate import get_learning_rate
from .optimizers import OPTIMIZERS
from ..dataclass import ModelParameter
from ..mtf_wrapper import (cast, einsum, equal, mod,
                           assign, assign_sub,
                           add, multiply, scoped, assign_add, identity, zeros_like, negative, rsqrt_eps,
                           optimizer_scalar)
from ..utils_mtf import feature_dims_used, get_fan_in

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def gradient_accumulation(ctx: OptimizerCtx):
    ctx.update_ops.append(assign_add(ctx.grad_buffer, add(ctx.grad, identity(ctx.grad_buffer))))


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
        ctx.grad = multiply(params.rezero_lr_multiplier, ctx.grad)

    features_used = feature_dims_used(params, var)
    large_tensor = features_used and len(var.shape.dims) > len(params.feature_dims)
    large_tensor |= not features_used and len(var.shape.dims) >= 2  # not norm or rezero + scalable catch-all
    large_tensor &= var.shape.size > 1  # not rezero
    large_tensor &= "norm" not in var.name  # not norm
    large_tensor &= "embed" not in var.name  # not input/output embedding, position embedding, attention map bias
    large_tensor &= "input" not in var.name or "lang_in" in var.name or "vid_in" in var.name  # not input
    large_tensor &= "output" not in var.name or "lang_out" in var.name or "vid_out" in var.name  # not output

    if large_tensor and params.weight_decay > 0:
        ctx.grad = add(ctx.grad, einsum([optimizer_scalar(params, params.weight_decay),
                                         cast(var.value, params.optimizer_calculation_dtype), learning_rate],
                                        output_shape=var.shape))

    if not large_tensor or not params.weight_standardisation:
        update_ops.append(assign_sub(var, ctx.grad))
        return

    val: mtf.Tensor = var.value - ctx.grad
    fan_in_size = np.prod([d.size for d in get_fan_in(params, var)])
    size = np.prod([d.size for d in var.shape.dims])
    max_fan = max(fan_in_size, size // fan_in_size)
    variance = ((1 - 1 / max_fan) / size ** 2 + 1 / max_fan - 2 / size + 1 / max_fan)
    if params.scale_by_depth:
        variance *= params.n_blocks
    val = einsum([val, rsqrt_eps(einsum([val, val, optimizer_scalar(params, 1 / val.size)], output_shape=[])),
                  optimizer_scalar(params, variance ** 0.5)], output_shape=var.shape)
    update_ops.append(assign(var, val))


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

    dtype = params.optimizer_calculation_dtype
    update_ops = []

    learning_rate_ctx = get_learning_rate(params, loss_list, update_ops)
    learning_rate = import_mtf(params, learning_rate_ctx.learning_rate, "learning_rate")

    step = cast(equal(mod(tf.cast(manual_step + 1, dtype),
                          import_mtf(params, params.grad_accumulation * 1., "grad_accum")),
                      import_mtf(params, 0., "zero")), dtype)
    neg_step = negative(step)
    mstep = add(1, neg_step)
    beta1 = add(1, multiply(neg_step, import_mtf(params, 1 - params.opt_beta1, "beta1")))
    beta2 = add(1, multiply(neg_step, import_mtf(params, 1 - params.opt_beta2, "beta2")))
    step_count = add(multiply(cast(learning_rate_ctx.global_steps_mtf, step.dtype), step), multiply(neg_step, 10 ** 9))

    ctx = OptimizerCtx(None, [], set(), {}, {}, params,
                       0, update_ops, {}, loss_list, {}, 0,
                       0, 0, mstep, step, neg_step, dtype, beta1, beta2,
                       learning_rate, step_count)
    for _ in gradients(ctx):
        full_name = f'{tf.get_variable_scope().name}/f"{ctx.var.name}/{params.optimizer}/grad_accumulation'
        if fn == "accumulate" or full_name in params.mesh.graph.name_to_variable:
            ctx.grad_buffer = variable(params, ctx.var, "grad_accumulation", ctx.var.shape)
        scoped(fn, gradient_accumulation if fn == "accumulate" else update, ctx)
    return params.mesh.graph.trainable_variables[0].graph.combine_assignments(update_ops), \
           learning_rate_ctx.learning_rate, {}
