import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import ModelParameter
from ..model import build
from ..mtf_wrapper import constant_scalar, divide
from ..optimizer import get_optimizer
from ..utils_core import reset_scope
from ..utils_mtf import WhileLoopWithControlDependencies

def get_train_model(params: ModelParameter):
    def train_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                    frame_mask_src, frame_mask_tag, token_mask, optimizer_mode="update"):

        reset_scope()
        params.variable_cache.clear()
        params.cached_parameters.clear()

        loss, loss_list, video_loss, accuracy, token_loss, frame_out, token_out = build(params,
                                                                                        frame_input,
                                                                                        cat_mask_src,
                                                                                        cat_mask_tag,
                                                                                        token_x_input,
                                                                                        token_y_input,
                                                                                        frame_mask_src,
                                                                                        frame_mask_tag,
                                                                                        token_mask)

        if params.multi_loss_strategy == "linear":
            loss_list = [loss]
        elif params.multi_loss_strategy == "mgda":
            loss_list = loss_list + [None]

        if optimizer_mode == "grad_return":
            variable_to_gradient = get_optimizer(loss_list, params, optimizer_mode)
            return frame_out, token_out, loss, video_loss, \
            token_loss, accuracy, variable_to_gradient

        update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, optimizer_mode)

        return frame_out, token_out, learning_rate, loss, video_loss, \
               token_loss, accuracy, update_ops, debug_gradients_dict

    def train_in_loop(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                    frame_mask_src, frame_mask_tag, token_mask):

        loop_input = [constant_scalar(params, 0, dtype=tf.int64),
                      constant_scalar(params, 0, dtype=tf.float32),
                      constant_scalar(params, 0, dtype=params.calculation_dtype),
                      constant_scalar(params, 0, dtype=params.calculation_dtype),
                      constant_scalar(params, 0, dtype=params.calculation_dtype)]

        grad_keys = []

        def body_fn(idx, loss, video_loss, token_loss, accuracy, *args):

            def inp_slice_fn(x: mtf.Tensor):
                if x is not None:
                    slice_dim = mtf.Dimension("batch_slice", params.macro_batching)
                    return mtf.gather(mtf.replace_dimensions(x, params.macro_batch_dim, [params.batch_dim, slice_dim]),
                        params.macro_batching, slice_dim)

                return x

            _frame_out, token_out, _loss, _video_loss, \
            _token_loss, _accuracy, variable_to_gradient = train_model(inp_slice_fn(frame_input),
                                                                       inp_slice_fn(cat_mask_src),
                                                                       inp_slice_fn(cat_mask_tag),
                                                                       inp_slice_fn(token_x_input),
                                                                       inp_slice_fn(token_y_input),
                                                                       inp_slice_fn(frame_mask_src),
                                                                       inp_slice_fn(frame_mask_tag),
                                                                       inp_slice_fn(token_mask),
                                                                       optimizer_mode="grad_return")

            if _loss is not None:
                loss = loss + _loss
            if _video_loss is not None:
                video_loss = video_loss + _video_loss
            if _token_loss is not None:
                token_loss = token_loss + _token_loss
            if _accuracy is not None:
                accuracy = accuracy + _accuracy

            grad_keys.extend(list(variable_to_gradient.keys()))

            return [mtf.add(idx, constant_scalar(params, 1, dtype=tf.int64), name="idx_add_one"), loss,
                    video_loss, token_loss, accuracy] + [variable_to_gradient[key] for key in grad_keys]

        def count_fn(idx, *args):
            return mtf.less(idx, params.grad_accumulation)

        loop_idx, loss,\
        video_loss,\
        token_loss,\
        accuracy,\
        *grads = WhileLoopWithControlDependencies(cond_fn=count_fn, body_fn=body_fn, inputs=loop_input,
                                                  has_accumulators=True).outputs

        grads = {k: divide(g, params.grad_accumulation) for k, g in zip(grad_keys, grads)}
        loss = divide(loss, params.grad_accumulation)
        video_loss = divide(video_loss, params.grad_accumulation)
        token_loss = divide(token_loss, params.grad_accumulation)
        accuracy = divide(accuracy, params.grad_accumulation)

        update_ops, learning_rate, debug_gradients_dict = get_optimizer([], params, "apply_grad",
                                                                        variable_to_gradient=grads)

        return None, None, learning_rate, loss, video_loss, token_loss, accuracy, [], debug_gradients_dict

    if params.grad_accumulation > 1:
        return train_in_loop

    return train_model
