import mesh_tensorflow as mtf
import tensorflow as tf

from src.dataclass import ModelParameter
from src.model import build
from src.optimizers import get_optimizer
from src.utils_core import _NAME_INDICES
from src.utils_mtf import WhileLoopWithControlDependencies


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

        update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, manual_global_step,
                                                                        "update")

        return frame_out, token_out, learning_rate, loss, video_loss, \
               token_loss, accuracy, update_ops, debug_gradients_dict

    if not params.split_grad_accumulation or params.batch_splits <= 1:
        return train_model

    def train_model_in_loop(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input,
                            frame_mask_src, frame_mask_tag, token_mask, manual_global_step):

        control_dependencies = []

        def model_loop_fn(idx):

            frame_out, \
            token_out, \
            learning_rate, \
            loss, \
            video_loss, \
            token_loss, \
            accuracy, \
            update_ops, \
            debug_gradients_dict = train_model(frame_input,
                                               cat_mask_src,
                                               cat_mask_tag,
                                               token_x_input,
                                               token_y_input,
                                               frame_mask_src,
                                               frame_mask_tag,
                                               token_mask,
                                               manual_global_step)

            for op in update_ops:
                control_dependencies.append(op)

            idx = mtf.identity(idx + mtf.constant(mesh=params.mesh, value=1, shape=[], dtype=tf.int32))

            return [idx]

        def count_fn(idx, *args):
            return mtf.less(idx, mtf.constant(params.mesh, (params.grad_accumulation - 1), shape=[], dtype=tf.int32))

        loop_input = [mtf.constant(mesh=params.mesh, value=0, shape=[], dtype=tf.int32)]
        loop = WhileLoopWithControlDependencies(cond_fn=count_fn, body_fn=model_loop_fn, inputs=loop_input,
                                                    control_dependencies=control_dependencies)
        loop_out = loop.outputs
        print(type(loop.graph), id(loop.graph))
        #loop.

        params.split_grad_accumulation = False
        return train_model(frame_input, cat_mask_src, cat_mask_tag, mtf.depend(token_x_input, [loop]), token_y_input,
                            frame_mask_src, frame_mask_tag, token_mask,  manual_global_step)


    return train_model_in_loop
