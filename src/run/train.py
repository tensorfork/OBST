import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import ModelParameter
from ..model import build
from ..mtf_wrapper import constant_scalar
from ..optimizer import get_optimizer
from ..utils_core import NAME_INDICES
from ..utils_mtf import unbind, deduplicate


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
        inputs = list(zip(*map(inp_slice_fn, inputs)))
        idx = constant_scalar(params, 0, dtype=params.optimizer_calculation_dtype)
        all_ops = []
        for i, args in enumerate(inputs, 1):
            NAME_INDICES.clear()
            loss, loss_list, video_loss, accuracy, token_loss, frame_out, token_out = build(params, *args)
            loss = none_cast(loss)
            video_loss = none_cast(video_loss)
            token_loss = none_cast(token_loss)
            if params.multi_loss_strategy == "linear":
                loss_list = [loss]
            elif params.multi_loss_strategy == "mgda":
                loss_list = [none_cast(x) for x in loss_list] + [None]
            graph: mtf.Graph = params.mesh.graph
            update_ops, learning_rate = get_optimizer(loss_list, params, idx, "update")
            if i != len(inputs):
                ops = graph.operations.copy()
                all_ops.extend([op for op in ops if not isinstance(op, mtf.Assign)])
                idx += 1
                for op in update_ops:
                    op: mtf.Assign = op
                    for var, inp in zip(op.variables, op.inputs):
                        var: mtf.Variable = var
                        inp = mtf.stop_gradient(mtf.cast(inp, var.activation_dtype))
                        var._outputs.clear()
                        var._outputs.append(inp)
                        inp._operation = var
                graph.operations.clear()
                graph.operations.extend([op for op in ops if isinstance(op, mtf.Variable)])
            else:
                ops = graph.operations.copy()
                graph.operations.clear()
                graph.operations.extend(all_ops)
                graph.operations.extend(ops)
                graph._all_variables = deduplicate(graph.all_variables)
                graph._trainable_variables = deduplicate(graph.trainable_variables)
                graph._operations = deduplicate(graph.operations)

        return frame_out, token_out, learning_rate, loss, video_loss, token_loss, accuracy, update_ops, {}

    return train_model
