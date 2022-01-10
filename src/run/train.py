import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import ModelParameter
from ..model import build
from ..mtf_wrapper import constant_scalar
from ..optimizer import get_optimizer
from ..utils_mtf import unbind


def none_cast(x: typing.Optional[mtf.Tensor]):
    if x is not None:
        return mtf.cast(x, tf.float64)


def get_train_model(params: ModelParameter):
    def train_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                    frame_mask_src, frame_mask_tag, token_mask, manual_global_step):
        slice_dim = mtf.Dimension("batch_slice", params.macro_batching)

        def inp_slice_fn(x: typing.Optional[mtf.Tensor]):
            if x is None:
                return [None] * params.macro_batching
            x = mtf.replace_dimensions(x, params.macro_batch_dim, [slice_dim, params.batch_dim])
            return unbind(x, slice_dim)

        inputs = (frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input, frame_mask_src, frame_mask_tag,
                  token_mask)
        inputs = zip(*map(inp_slice_fn, inputs))
        idx = constant_scalar(params, 0, dtype=tf.int64)
        for args in inputs:
            mtf.depend()
            loss, loss_list, video_loss, accuracy, token_loss, frame_out, token_out = build(params, *args)
            loss = none_cast(loss)
            video_loss = none_cast(video_loss)
            token_loss = none_cast(token_loss)
            if params.multi_loss_strategy == "linear":
                loss_list = [loss]
            elif params.multi_loss_strategy == "mgda":
                loss_list = [none_cast(x) for x in loss_list] + [None]

            update_ops, learning_rate = get_optimizer(loss_list, params, idx, "update")
            idx += 1
            for op in update_ops:
                for var in op.variables:
                    mtf.depend(var, op)
        return frame_out, token_out, learning_rate, loss, video_loss, token_loss, accuracy, update_ops, {}

    return train_model
