"""
Stores custom optimizer classes as well as a custom optimizer creation utility as a handy wrapper
b"""

import typing

import mesh_tensorflow as mtf
import tensorflow as tf2

from .backend import import_mtf, import_float, get_var, variable
from .backend import variable
from .context import OptimizerCtx
from .context import OptimizerCtx
from .gradients import MULTI_LOSS_GRADIENTS
from .learning_rate import get_learning_rate
from .optimizers import OPTIMIZERS
from ..dataclass import ModelParameter
from ..model.embedding import Gather
from ..mtf_wrapper import (cast, constant_float, constant_scalar, einsum, equal, greater_equal, mod, reduce_sum, assign,
                           add, multiply, scoped, identity, zeros_like, negative, optimizer_scalar, reciprocal,
                           reduce_mean, broadcast)
from ..utils_mtf import feature_dims_used, to_fp32, gradient_iterator, assign_sub, assign_add

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def gradient_accumulation(ctx: OptimizerCtx):
    ctx.update_ops.append(assign_add(ctx.op, ctx.grad_buffer, add(ctx.grad, identity(ctx.grad_buffer))))


def update(ctx: OptimizerCtx):
    params = ctx.params
    update_ops = ctx.update_ops
    learning_rate = ctx.learning_rate

    var = ctx.var
    if ctx.grad_buffer is not None:
        ctx.grad = reduce_mean(broadcast(identity(ctx.grad_buffer.value), [params.batch_dim] + ctx.grad.shape.dims),
                               params.batch_dim)
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
    large_tensor &= "rezero" not in var.name  # not norm
    large_tensor &= "embed" not in var.name  # not input/output embedding, position embedding, attention map bias
    large_tensor &= "input" not in var.name or "lang_in" in var.name or "vid_in" in var.name  # not input
    large_tensor &= "output" not in var.name or "lang_out" in var.name or "vid_out" in var.name  # not output

    momentum = variable(params, ctx.var, "momentum", ctx.var.shape)
    update_ops.append(assign_add(ctx.op, momentum.operation, ctx.grad * (1 - params.momentum), params.momentum))

    if large_tensor and params.weight_decay > 0:
        momentum += einsum([cast(var.value, params.optimizer_calculation_dtype), learning_rate,
                            optimizer_scalar(params, params.weight_decay)], output_shape=var.shape)

    update_ops.append(assign_sub(ctx.op, ctx.var, momentum))


def get_optimizer(loss_list: typing.List[mtf.Tensor], params: ModelParameter, manual_step: mtf.Tensor, fn: str
                  ) -> typing.Tuple[typing.List[mtf.Assign], mtf.Tensor]:
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

    step = cast(equal(mod(cast(manual_step + optimizer_scalar(params, 1), dtype),
                          import_mtf(params, params.grad_accumulation * 1., "grad_accum")),
                      import_mtf(params, 0., "zero")), dtype)
    neg_step = -step
    mstep = 1 + neg_step
    beta1 = 1 + neg_step * import_mtf(params, 1 - params.opt_beta1, "beta1")
    beta2 = 1 + neg_step * import_mtf(params, 1 - params.opt_beta2, "beta2")
    step_count = cast(learning_rate_ctx.global_steps_mtf, step.dtype) * step + mstep * 10 ** 9 + 1

    first_grad = {}
    loss_1__loss_1 = loss_1__loss_2 = loss_2__loss_2 = 0
    mgda = params.multi_loss_strategy == "mgda"

    if mgda:
        loss_1__loss_1 = constant_float(params, 0, shape=[params.head_dim])
        loss_1__loss_2 = constant_float(params, 0, shape=[params.head_dim])
        loss_2__loss_2 = constant_float(params, 0, shape=[params.head_dim])

    tensor_to_gradient: typing.Dict[mtf.Tensor, typing.List[int, int, mtf.Tensor, mtf.Operation]] = {}
    tensor_to_var = {}

    for loss_idx, loss in enumerate(loss_list):
        if mgda and loss_idx == 2:
            v1v1 = reduce_sum(loss_1__loss_1, output_shape=[])
            v1v2 = reduce_sum(loss_1__loss_2, output_shape=[])
            v2v2 = reduce_sum(loss_2__loss_2, output_shape=[])
            min_gamma = 0.001
            gamma = multiply(constant_float(params, value=(1 - min_gamma), shape=[]),
                             to_fp32(greater_equal(v1v2, v1v1)))
            gamma += einsum([constant_float(params, value=min_gamma, shape=[]), to_fp32(greater_equal(v1v2, v2v2)),
                             to_fp32(equal(gamma, 0))], output_shape=[])
            gamma += einsum([optimizer_scalar(params, -1),
                             to_fp32(equal(gamma, 0)),
                             add(v1v2, negative(v2v2)),
                             reciprocal(add(v1v1, v2v2) - multiply(-2, v1v2))],
                            output_shape=[])

            loss = loss_list[0] * gamma + loss_list[1] * (1 - gamma)

        operations = loss.graph.operations
        xs = [x.outputs[0] for x in params.mesh.graph.trainable_variables]
        tensor_to_var = dict(zip(xs, params.mesh.graph.trainable_variables))
        loss_grad = constant_scalar(params, 1.0)
        downstream = set(xs)

        for op in operations:
            if op.has_gradient and (set(op.inputs) & downstream):
                downstream |= set(op.outputs)

        tensor_to_gradient: typing.Dict[mtf.Tensor, typing.List[int, int, mtf.Tensor,
                                                                mtf.Operation]] = {loss: [0, 0, loss_grad, None]}

        with tf.variable_scope(loss.graph.captured_variable_scope):
            for op in operations[::-1]:
                grad_outputs = []
                for out in op.outputs:
                    if out not in tensor_to_gradient:
                        grad_outputs.append(None)
                        continue

                    grad_list: typing.Tuple[int, int, mtf.Tensor, mtf.Operation] = tensor_to_gradient[out]
                    grad_outputs.append(grad_list[2])
                    grad_list[0] += 1

                if not op.has_gradient or not any(grad_outputs) or not (set(op.inputs) & downstream):
                    continue
                for inner_op, inp, grad in gradient_iterator(params, op, grad_outputs):
                    if inp not in downstream or grad is None:
                        continue

                    if inp in tensor_to_gradient:
                        grad_list = tensor_to_gradient[inp]
                        grad_list[1] += 1
                        grad_list[2] += grad
                        grad_list[3] = inner_op
                    else:
                        tensor_to_gradient[inp] = grad_list = [0, 1, grad, inner_op]

                    if isinstance(inner_op, Gather):
                        var: mtf.Variable = tensor_to_var[inp]

                        ctx = OptimizerCtx(op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params,
                                           loss_idx, update_ops, {}, loss_list, first_grad,
                                           loss_1__loss_1, loss_1__loss_2, loss_2__loss_2, mstep, step, neg_step, dtype,
                                           beta1, beta2, learning_rate, step_count)

                        ctx.variable_to_gradient[var] = grad = cast(grad_list[2], params.optimizer_calculation_dtype)
                        scoped(fn, update, ctx(inp, var, grad))

    ctx = OptimizerCtx(op, grad_outputs, downstream, tensor_to_gradient, tensor_to_var, params,
                       loss_idx, update_ops, {}, loss_list, first_grad,
                       loss_1__loss_1, loss_1__loss_2, loss_2__loss_2, mstep, step, neg_step, dtype,
                       beta1, beta2, learning_rate, step_count)
    ctx.variable_to_gradient = {var: cast(tensor_to_gradient[tensor][2], params.optimizer_calculation_dtype)
                                for tensor, var in tensor_to_var.items() if var not in ctx.variable_to_gradient}
    for tensor, var in tensor_to_var.items():
        scoped(fn, update, ctx(tensor, var, ctx.variable_to_gradient[var]))

    return ctx.update_ops, learning_rate
