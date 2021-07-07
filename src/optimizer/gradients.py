import mesh_tensorflow as mtf
import tensorflow as tf2

from .context import OptimizerCtx
from ..mtf_wrapper import add_n, einsum, add, negative, minimum, gradients as mtf_gradients

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
    g_square = [add(1e-8, einsum([g, g], output_shape=[])) for g in all_grads[1:]]

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
    params = ctx.params
    loss_list = ctx.loss_list
    variables = [v.outputs[0] for v in params.mesh.graph.trainable_variables]
    for var, grad in zip(variables, mtf_gradients(loss_list, variables)):
        ctx(var, grad)
        yield None
