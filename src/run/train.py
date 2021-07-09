import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import ModelParameter
from ..model import build
from ..mtf_wrapper import constant_scalar
from ..utils_mtf import WhileLoopWithControlDependencies
from ..optimizer import get_optimizer


def get_train_model(params: ModelParameter):
    def train_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                    frame_mask_src, frame_mask_tag, token_mask, manual_global_step):

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

        update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, manual_global_step, "update")

        return frame_out, token_out, learning_rate, loss, video_loss, \
               token_loss, accuracy, update_ops, debug_gradients_dict

    def train_in_loop(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                    frame_mask_src, frame_mask_tag, token_mask, manual_global_step):

        loop_input = [constant_scalar(params, 0, dtype=tf.int32),
                      constant_scalar(params, 0, dtype=tf.float32),
                      constant_scalar(params, 0, dtype=params.calculation_dtype),
                      constant_scalar(params, 0, dtype=params.calculation_dtype),
                      constant_scalar(params, 0, dtype=params.calculation_dtype),
                      constant_scalar(params, 0, dtype=params.calculation_dtype)]

        def body_fn(idx, loss, learning_rate, video_loss, token_loss, accuracy):

            '''
            frame_out, token_out, _learning_rate, _loss, _video_loss, \
            _token_loss, _accuracy, update_ops, debug_gradients_dict = train_model(frame_input,
                                                                                    cat_mask_src,
                                                                                    cat_mask_tag,
                                                                                    token_x_input,
                                                                                    token_y_input,
                                                                                    frame_mask_src,
                                                                                    frame_mask_tag,
                                                                                    token_mask,
                                                                                    mtf.identity(idx))
            '''

            _loss, loss_list, _video_loss, _accuracy, _token_loss, frame_out, token_out = build(params,
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

            update_ops, _learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, mtf.identity(idx),
                                                                            "update")

            print('update_ops', update_ops)

            if _loss is not None:
                loss = _loss
            if _learning_rate is not None:
                learning_rate = _learning_rate
            if _video_loss is not None:
                video_loss = _video_loss
            if _token_loss is not None:
                token_loss = _token_loss
            if _accuracy is not None:
                accuracy = _accuracy

            return {"outputs": [mtf.add(idx, 1, name="idx_add_one"), loss,
                    learning_rate, video_loss, token_loss, accuracy],
                    "control_dependencies": update_ops}

            #return mtf.add(idx, 1, name="idx_add_one"), loss, learning_rate, video_loss, token_loss, accuracy

        def count_fn(idx, *args):
            return mtf.less(idx, constant_scalar(params, params.macro_batching, dtype=tf.int32))

        _, loss,\
        learning_rate,\
        video_loss,\
        token_loss,\
        accuracy = WhileLoopWithControlDependencies(cond_fn=count_fn, body_fn=body_fn, inputs=loop_input).outputs




        return None, None, learning_rate, loss, video_loss, token_loss, accuracy, [], None

    if params.macro_batching > 1:
        return train_in_loop

    return train_model
