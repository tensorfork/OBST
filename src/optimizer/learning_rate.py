import typing

import mesh_tensorflow as mtf
import tensorflow as tf2

from .backend import import_float, import_fully_replicated
from .. import tf_wrapper as tfw
from ..dataclass import ModelParameter, LearningRateConfig
from ..utils_mtf import weighted_add

tf = tf2.compat.v1


class LearningRateCtx:
    def __init__(self, params: ModelParameter, loss_list: typing.List[mtf.Tensor],
                 update_ops: typing.List[mtf.Operation]):
        global_step = tf.train.get_or_create_global_step()
        self.params = params
        self.learning_rate = tf.constant(value=params.learning_rate, shape=[], dtype=tf.float32)
        self.global_steps_float = tf.cast(global_step, tf.float32)
        self.loss_list = loss_list
        self.global_steps_mtf = import_fully_replicated(params, global_step, [], "mtf_learning_rate")
        self.update_ops = update_ops
        self.config: typing.Optional[LearningRateConfig] = None


def linear_warmup(ctx: LearningRateCtx):
    warmup_steps_float = import_float(ctx.config.final_step)
    is_warmup = tfw.cast(ctx.global_steps_float < warmup_steps_float, tf.float32)
    warmup_factor = weighted_add(tfw.divide(ctx.global_steps_float, warmup_steps_float), 1, is_warmup)
    ctx.learning_rate = tfw.multiply(ctx.learning_rate, warmup_factor)


def exponential_decay(ctx: LearningRateCtx):
    base = import_float(ctx.config.factor)
    exp = tfw.maximum(tfw.subtract(ctx.global_steps_float, import_float(ctx.config.start_step), import_float(0)))
    decay = tfw.pow(base, exp)
    ctx.learning_rate = tfw.multiply(ctx.learning_rate, decay)


def linear_decay(ctx: LearningRateCtx):
    start_step = import_float(ctx.config.start_step)
    final_step = import_float(ctx.config.final_step)
    current_step = tfw.subtract(ctx.global_steps_float, start_step)
    final_step = tfw.subtract(final_step, start_step)
    decay = tfw.subtract(1, tfw.divide(current_step, final_step))
    decay = tfw.maximum(tfw.minimum(decay, 1), 0)
    ctx.learning_rate = tfw.multiply(ctx.learning_rate, decay)


def lower_bound(ctx: LearningRateCtx):
    ctx.learning_rate = tfw.maximum(ctx.learning_rate, ctx.config.factor)


def upper_bound(ctx: LearningRateCtx):
    ctx.learning_rate = tfw.minimum(ctx.learning_rate, ctx.config.factor)


MODULES = {"linear_warmup": linear_warmup,
           "exponential_decay": exponential_decay,
           "linear_decay": linear_decay,
           "lower_bound": lower_bound,
           "upper_bound": upper_bound}


def get_learning_rate(params: ModelParameter, loss_list: typing.List[mtf.Tensor],
                      update_ops: typing.List[mtf.Operation]) -> LearningRateCtx:
    ctx = LearningRateCtx(params, loss_list, update_ops)
    for name, keys in params.learning_rate_config.items():
        ctx.config = keys
        tfw.scoped(name, MODULES[name], ctx)
    return ctx
