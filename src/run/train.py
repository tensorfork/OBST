import mesh_tensorflow as mtf
import tensorflow as tf

from src.dataclass import ModelParameter
from src.model import build
from src.optimizers import get_optimizer
from src.utils_core import _NAME_INDICES


def get_train_model(params: ModelParameter):
    def train_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                   frame_mask_src, frame_mask_tag, token_mask, manual_global_step):

        _NAME_INDICES.clear()

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

        update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, manual_global_step)

        return frame_out, token_out, learning_rate, loss, video_loss, \
               token_loss, accuracy, update_ops, debug_gradients_dict

    if not params.split_grad_accumulation or params.batch_splits <= 1:
        return train_model

    def train_model_in_loop(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                            frame_mask_src, frame_mask_tag, token_mask, manual_global_step):

        def model_loop_fn(idx, frame_input, cat_mask_src, cat_mask_tag,
                          token_x_input, token_y_input, frame_mask_src,
                          frame_mask_tag, token_mask, *args):

            frame_out, token_out, learning_rate, loss, video_loss, \
            token_loss, accuracy, update_ops, debug_gradients_dict = train_model(frame_input,
                                                                                 cat_mask_src,
                                                                                 cat_mask_tag,
                                                                                 token_x_input,
                                                                                 token_y_input,
                                                                                 frame_mask_src,
                                                                                 frame_mask_tag,
                                                                                 token_mask,
                                                                                 manual_global_step)

            if frame_out is None:
                frame_out = args[0]
            if token_out is None:
                token_out = args[1]
            if video_loss is None:
                video_loss = args[2]
            if token_loss is None:
                token_loss = args[3]
            if accuracy is None:
                accuracy = args[4]

            print(update_ops)
            with tf.control_dependencies(update_ops):
                idx = mtf.identity(idx)
                idx = idx + 1

                return [idx, frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input, frame_mask_src,
                        frame_mask_tag, token_mask, manual_global_step] + \
                       [frame_out, token_out, loss, video_loss, token_loss, accuracy]

        def count_fn(idx, *args):
            return mtf.less(idx, mtf.constant(params.mesh, (params.grad_accumulation - 1), shape=[], dtype=tf.int32))

        loop_input = [mtf.zeros(mesh=params.mesh, shape=[], dtype=tf.int32)]

        if frame_input is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.uint32))
        else:
            loop_input.append(frame_input)

        if cat_mask_src is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(cat_mask_src)

        if cat_mask_tag is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(cat_mask_tag)

        if token_x_input is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(token_x_input)

        if token_y_input is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(token_y_input)

        if frame_mask_src is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(frame_mask_src)

        if frame_mask_tag is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(frame_mask_tag)

        if token_mask is None:
            loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.int32))
        else:
            loop_input.append(token_mask)

        #loop_input = loop_input + [frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
        #                    frame_mask_src, frame_mask_tag, token_mask, manual_global_step]

        loop_input.append(mtf.constant(params.mesh, value=0,
                                       shape=[params.batch_dim, params.sequence_dim]
                                             + params.frame_input_shape.dims[2:], dtype=tf.uint32))
        loop_input.append(mtf.constant(params.mesh, value=0, shape=mtf.Shape([params.batch_dim,
                                                                              params.sequence_dim,
                                                                              params.token_dim_shape[-1],
                                                                              params.vocab_dim]), dtype=tf.float32))
        loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.float32))
        loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.float32))
        loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.float32))
        loop_input.append(mtf.constant(params.mesh, value=0, shape=[], dtype=tf.float32))

        loop_out = mtf.while_loop(cond_fn=count_fn, body_fn=model_loop_fn, inputs=loop_input)

        with tf.control_dependencies(loop_out):

            params.split_grad_accumulation = False

            return train_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                               frame_mask_src, frame_mask_tag, token_mask,  manual_global_step)


    return train_model_in_loop
