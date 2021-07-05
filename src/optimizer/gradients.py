import mesh_tensorflow as mtf
import tensorflow as tf2

from .backend import variable
from .context import OptimizerCtx
from ..model.revnet import RevGradOp
from ..mtf_wrapper import (add_n, einsum, assign, add, negative, reshape, minimum)

tf = tf2.compat.v1
zeros = tf.zeros_initializer()


def pcgrad(ctx: OptimizerCtx, grad: mtf.Tensor):
    op = ctx.op
    loss_idx = ctx.loss_idx
    loss_list = ctx.loss_list
    first_grad = ctx.first_grad

    if 'body' not in op.name:
        return grad

    if loss_idx < len(loss_list) - 1:
        first_grad[op.name] = grad
        return None

    all_grads = [grad, first_grad[op.name]]
    g_square = [add(1e-6, einsum([g, g], output_shape=[])) for g in all_grads[1:]]

    for i in range(len(all_grads)):
        grad = all_grads.pop(0)
        for g, sq in zip(all_grads, g_square):
            grad = add(grad,
                       negative(einsum([g, minimum(einsum([grad, g], output_shape=[]), 0), sq],
                                       output_shape=grad.shape)))

        all_grads.append(grad)
        g_square.append(einsum([g, g], output_shape=[]))

    return add_n(all_grads)


def mgda(ctx: OptimizerCtx, grad: mtf.Tensor):
    op = ctx.op
    loss_idx = ctx.loss_idx
    first_grad = ctx.first_grad
    params = ctx.params

    if loss_idx == 2:
        return None

    if 'body' not in op.name:
        return grad

    if loss_idx == 0:
        first_grad[op.name] = grad
        return None

    elif loss_idx == 1:
        ctx.loss_1__loss_1 = add(ctx.loss_1__loss_1,
                                 einsum([first_grad[op.name], first_grad[op.name]], [params.head_dim]))
        ctx.loss_1__loss_2 = add(ctx.loss_1__loss_2,
                                 einsum([first_grad[op.name], grad], [params.head_dim]))
        ctx.loss_2__loss_2 = add(ctx.loss_2__loss_2, einsum([grad, grad], [params.head_dim]))

        del first_grad[op.name]
        return None

    return grad


MULTI_LOSS_GRADIENTS = {'mgda': mgda,
                        'pcgrad': pcgrad}


def gradients(ctx: OptimizerCtx):
    op = ctx.op
    grad_outputs = ctx.grad_outputs
    downstream = ctx.downstream
    tensor_to_gradient = ctx.tensor_to_gradient
    tensor_to_var = ctx.tensor_to_var
    params = ctx.params
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
            grad_list[2] = add(grad, grad_list[2])
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
            grad = MULTI_LOSS_GRADIENTS[params.multi_loss_strategy](ctx, grad)

        if grad is None:
            continue

        ctx(var, grad)
        yield None
