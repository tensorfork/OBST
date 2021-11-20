import mesh_tensorflow as mtf

from .context import OptimizerCtx
from ..mtf_wrapper import (cast, optimizer_scalar, einsum, greater, minimum,
                           reduce_mean, reduce_sum, add, multiply, maximum, reciprocal, square,
                           rsqrt, sqrt, add_n)
from ..utils_mtf import weighted_add


def adaptive_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    grd_norm = maximum(sqrt(einsum([ctx.grad, ctx.grad], output_shape=[])), 1e-6)
    wgt_norm = maximum(sqrt(einsum([cast(ctx.var.value, ctx.params.optimizer_calculation_dtype)] * 2, output_shape=[])),
                       0.001)
    return weighted_add(einsum([wgt_norm, reciprocal(grd_norm), optimizer_scalar(ctx.params, gradient_clip), ctx.grad],
                               output_shape=ctx.grad.shape), ctx.grad,
                        cast(greater(multiply(grd_norm, reciprocal(wgt_norm)), gradient_clip),
                             ctx.params.optimizer_calculation_dtype))


def l2norm_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return einsum([ctx.grad, optimizer_scalar(ctx.params, gradient_clip),
                   rsqrt(maximum(einsum([ctx.grad, ctx.grad], []), gradient_clip ** -2))])


def global_l2norm_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    if ctx.global_norm_reciprocal is None:
        global_sum = add_n([reduce_sum(square(grad)) for grad in ctx.variable_to_gradient.values()])
        ctx.global_norm_reciprocal = rsqrt(maximum(global_sum, gradient_clip ** -2))
    return einsum([ctx.grad, optimizer_scalar(ctx.params, gradient_clip), ctx.global_norm_reciprocal])


def value_gradient_clipping(ctx: OptimizerCtx, gradient_clip: str) -> mtf.Tensor:
    gradient_clip = float(gradient_clip)
    return maximum(minimum(ctx.grad, gradient_clip), -gradient_clip)


def gradient_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return ctx.grad - reduce_mean(ctx.grad)


def weight_centralisation(ctx: OptimizerCtx) -> mtf.Tensor:
    return add(ctx.grad, reduce_mean(ctx.var.value))


def multiply_learning_rate(ctx: OptimizerCtx) -> mtf.Tensor:
    return multiply(ctx.grad, ctx.learning_rate)


OPTIMIZERS = {"adaptive_clip": adaptive_gradient_clipping,
              "l2norm_clip": l2norm_gradient_clipping,
              "value_clip": value_gradient_clipping,
              "gradient_centralisation": gradient_centralisation,
              "weight_centralisation": weight_centralisation,
              "learning_rate": multiply_learning_rate,
              "global_l2norm_clip": global_l2norm_gradient_clipping,
              }


def graft(ctx: OptimizerCtx, optimizer: str, *params: str) -> mtf.Tensor:
    return einsum([ctx.grad, rsqrt(reduce_sum(square(ctx.grad))),
                   sqrt(reduce_sum(square(OPTIMIZERS[optimizer](ctx, *params))))],
                  output_shape=ctx.grad.shape)


OPTIMIZERS['graft'] = graft
