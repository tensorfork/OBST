from ..dataclass import ModelParameter
from ..model import build
from ..optimizers import get_optimizer


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

        update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, manual_global_step,
                                                                        "update")

        return frame_out, token_out, learning_rate, loss, video_loss, \
               token_loss, accuracy, update_ops, debug_gradients_dict

    return train_model
