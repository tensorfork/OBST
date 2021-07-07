import mesh_tensorflow as mtf

from .backend import variable
from .context import OptimizerCtx
from ..mtf_wrapper import (cast, optimizer_scalar, einsum, greater, minimum,
                           reduce_mean, reduce_sum, assign, add, multiply, maximum, sqrt_eps, rsqrt_eps,
                           reciprocal, square, reduce_max, rsqrt, sqrt, pow, negative)
from ..utils_mtf import weighted_add, get_fan_in


def opt_rsqrt(tensor: mtf.Tensor):
    return reciprocal(add(sqrt(tensor), 1e-8))


def adam(ctx: OptimizerCtx) -> mtf.Tensor:
    exp_avg_p2_ptr = variable(ctx.params, ctx.var, 'exp_avg_p2', ctx.var.shape)
    exp_avg_p1_ptr = variable(ctx.params, ctx.var, 'exp_avg_p1', ctx.var.shape)

    exp_avg_p2 = multiply(weighted_add(exp_avg_p2_ptr, square(ctx.grad), ctx.beta2),
                          reciprocal(add(1, negative(pow(ctx.beta2, ctx.step_count)))))
    ctx.grad = multiply(weighted_add(exp_avg_p1_ptr, ctx.grad, ctx.beta1),
                        reciprocal(add(1, negative(pow(ctx.beta1, ctx.step_count)))))

    ctx.update_ops.append(assign(exp_avg_p2_ptr, exp_avg_p2))
    ctx.update_ops.append(assign(exp_avg_p1_ptr, ctx.grad))
    return multiply(ctx.grad, opt_rsqrt(exp_avg_p2))


def novograd(ctx: OptimizerCtx) -> mtf.Tensor:
    exp_avg_p1 = exp_avg_p1_ptr = variable(ctx.params, ctx.var, "exp_avg_p1", ctx.var.shape)
    exp_avg_p2 = exp_avg_p2_ptr = variable(ctx.params, ctx.var, "exp_avg_p2", [])

    exp_avg_p2 = weighted_add(exp_avg_p2, reduce_sum(square(ctx.grad)), ctx.beta2)
    ctx.update_ops.extend([assign(exp_avg_p1_ptr, add(multiply(ctx.beta1, exp_avg_p1_ptr),
                                                      multiply(ctx.grad, opt_rsqrt(exp_avg_p2)))),
                           assign(exp_avg_p2_ptr, exp_avg_p2)])
    return add(multiply(ctx.beta1, exp_avg_p1), multiply(ctx.grad, opt_rsqrt(exp_avg_p2)))


def sm3(ctx: OptimizerCtx) -> mtf.Tensor:
    weight_update = variable(ctx.params, ctx.var, "dim0", [ctx.var.shape.dims[0]])
    buffer = [weight_update]

    for i in range(1, ctx.var.shape.ndims):
        buffer.append(variable(ctx.params, ctx.var, f"dim{i}", [ctx.var.shape.dims[i]]))
        weight_update = minimum(weight_update, buffer[-1])

    weight_update = add(weight_update, square(ctx.grad))

    ctx.update_ops.extend([assign(buf_ptr, reduce_max(weight_update, output_shape=[dim]))
                           for buf_ptr, dim in zip(buffer, weight_update.shape.dims)])
    return multiply(ctx.grad, opt_rsqrt(weight_update))


def return_grad(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad


def adaptive_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    grd_norm = sqrt_eps(einsum([ctx.grad, ctx.grad], reduced_dims=get_fan_in(ctx.params, ctx.var)))
    wgt_norm = sqrt_eps(einsum([ctx.var.value, ctx.var.value], reduced_dims=get_fan_in(ctx.params, ctx.var)))
    return weighted_add(einsum([grd_norm, reciprocal(wgt_norm), optimizer_scalar(ctx.params, gradient_clip), ctx.grad],
                               output_shape=ctx.grad.shape), ctx.grad,
                        cast(greater(multiply(wgt_norm, reciprocal(grd_norm)), gradient_clip),
                             ctx.params.optimizer_calculation_dtype))


def l2norm_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return einsum([ctx.grad, optimizer_scalar(ctx.params, gradient_clip),
                   rsqrt(maximum(einsum([ctx.grad, ctx.grad], []), gradient_clip ** -2))])


def value_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return maximum(minimum(ctx.grad, gradient_clip), -gradient_clip)


def gradient_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad - reduce_mean(ctx.grad)


def weight_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return add(ctx.grad, reduce_mean(ctx.var.value))


def multiply_learning_rate(ctx: OptimizerCtx) -> mtf.Tensor:
    return multiply(ctx.grad, ctx.learning_rate)


OPTIMIZERS = {"adam": adam,
              "sm3": sm3,
              "novograd": novograd,
              "sgd": return_grad,
              "adaptive_clip": adaptive_gradient_clipping,
              "l2norm_clip": l2norm_gradient_clipping,
              "value_clip": value_gradient_clipping,
              "gradient_centralisation": gradient_centralisation,
              "weight_centralisation": weight_centralisation,
              "learning_rate": multiply_learning_rate
              }
