import mesh_tensorflow as mtf
import tensorflow as tf

from ..dataclass import ModelParameter
from ..model import build
from ..mtf_wrapper import (constant_scalar, log, argmax, reshape, one_hot, equal, less_equal, mtf_range, greater,
                           reduce_sum, cast, shift, ones, zeros, constant, random_uniform, greater_equal, logical_not,
                           anonymize, add, negative, multiply)
from ..utils_mtf import concat, pad, utils_slice, to_fp32, weighted_add

tf1 = tf.compat.v1
Dataset = tf1.data.Dataset


def autoregressive_model(params: ModelParameter,
                         frame_input=None, token_x_input=None, token_y_input=None,
                         frame_mask_src=None, frame_mask_tag=None, token_mask=None,
                         initial_pos=None, sampling_temperature=None, end_iterations=None):
    if params.use_video:
        # todo: fix token shift for video (Jan).
        tkn_per_frame = mtf.Dimension("language_token_per_frame",
                                      params.language_token_per_frame)
        shape = [params.batch_dim, params.sequence_dim, tkn_per_frame, params.vocab_dim]

        def body_fn(position, token_x_input, token_y_input, frame_input,
                    frame_mask_src, frame_mask_tag, token_mask, *states):

            _, _, _, _, _, frame_out, token_out = build(params,
                                                        frame_input,
                                                        ones(params.mesh, [], tf.float32),
                                                        ones(params.mesh, [], tf.float32),
                                                        token_x_input,
                                                        token_y_input,
                                                        frame_mask_src,
                                                        frame_mask_tag,
                                                        token_mask)

            frame_input = weighted_add(pad(frame_out, params.sequence_dim, (0, 1)), frame_input,
                                       one_hot(position, params.frame_input_sequence, dtype=tf.float32))

            if params.use_language:
                one_hot_sequence = one_hot(position, params.sequence_dim, dtype=tf.float32)
                token_out = argmax(reshape(token_out, new_shape=shape), params.vocab_dim)
                padding_token = to_fp32(equal(token_out, params.padding_token))

                token_x_input = weighted_add(reshape(token_out, new_shape=params.token_dim_shape),
                                             token_x_input,
                                             one_hot(position, params.sequence_dim, dtype=tf.int32))

                token_pad = less_equal(mtf_range(params.mesh, tkn_per_frame, dtype=tf.float32),
                                       to_fp32(argmax(padding_token, reduced_dim=tkn_per_frame)),
                                       output_shape=token_out.shape)

                token_mask = weighted_add(reshape(to_fp32(token_pad), new_shape=params.token_dim_shape),
                                          to_fp32(token_mask), one_hot_sequence)

                frame_pad = to_fp32(greater(reduce_sum(padding_token, reduced_dim=tkn_per_frame), 0))
                token_x_input = weighted_add(frame_pad, to_fp32(token_x_input), one_hot_sequence)

                token_x_input = cast(token_x_input, dtype=tf.int32)

            return add(position, 1), token_x_input, token_y_input, frame_input, frame_mask_src, \
                   frame_mask_tag, token_mask

        if token_mask is not None:
            token_mask = to_fp32(token_mask)
        if frame_mask_src is not None:
            frame_mask_src = to_fp32(frame_mask_src)
        if frame_mask_tag is not None:
            frame_mask_tag = to_fp32(frame_mask_tag)

        while_loop_inputs = [zeros(params.mesh, [], tf.int32) + params.initial_autoregressive_position,
                             token_x_input, token_y_input, frame_input, frame_mask_src, frame_mask_tag,
                             token_mask]

    else:  # -> params.use_language
        def body_fn(position, token_x, token_y, sampling_temperature, *states):
            _, _, _, _, _, _, token_out = build(params,
                                                ones(params.mesh, [], tf.float32),
                                                ones(params.mesh, [], tf.float32),
                                                ones(params.mesh, [], tf.float32),
                                                token_x,
                                                token_y,
                                                ones(params.mesh, [], tf.float32),
                                                ones(params.mesh, [], tf.float32),
                                                ones(params.mesh, [], tf.float32))

            one_hot_mask = one_hot(position, output_dim=params.sequence_dim, dtype=tf.int32)
            token_out = add(cast(token_out, dtype=tf.float32),
                            multiply(log(negative(log(random_uniform(params, token_out.shape,
                                                                     maxval=1, minval=1e-9, dtype=tf.float32)))),
                                     negative(sampling_temperature)))
            token_out = argmax(token_out, params.vocab_dim)

            token_out = shift(token_out, offset=1, dim=params.sequence_dim, wrap=False)

            return (add(position, 1), weighted_add(token_out, token_x, one_hot_mask),
                    token_y, cast(sampling_temperature, dtype=tf.float32))

        if initial_pos is None:
            initial_pos = constant(params, value=params.initial_autoregressive_position, dtype=tf.int32)

        if params.debug_sample:
            token_initial_pos_mask = less_equal(mtf_range(params.mesh, params.sequence_dim, dtype=tf.int32),
                                                initial_pos)
            token_initial_pos_mask = cast(token_initial_pos_mask, tf.int32)
            token_x_input_a = utils_slice(token_x_input, 0, 1, dim=params.batch_dim)
            token_x_input_b = multiply(token_x_input_a, token_initial_pos_mask)
            token_x_input = concat([token_x_input_a, token_x_input_b], dim=token_x_input_a.shape[0])

        if sampling_temperature is None:
            sampling_temperature = constant_scalar(params, params.sampling_temperature, dtype=tf.float32)

        if end_iterations is None:
            end_iterations = constant(params, value=params.sequence_length, dtype=tf.int32)

        while_loop_inputs = [initial_pos, token_x_input, token_y_input, sampling_temperature]

    def cond_fn(position, *states):
        is_done = greater_equal(position, end_iterations)
        is_done = reduce_sum(is_done)

        return logical_not(is_done)

    loop_out = mtf.while_loop(cond_fn=cond_fn, body_fn=body_fn, inputs=while_loop_inputs)

    token_out = None
    frame_out = None
    if params.use_language:
        token_out = loop_out[1]
    if params.use_video:
        frame_out = loop_out[3]

    return token_out, frame_out


def get_infrence_model(params: ModelParameter):
    def infrence_model(frame_input, cat_mask_src, cat_mask_tag, token_x_input, token_y_input, frame_mask_src,
                       frame_mask_tag,
                       token_mask, initial_pos, sampling_temperature, end_iterations):

        if params.use_autoregressive_sampling:
            token_out, frame_out = autoregressive_model(params,
                                                        frame_input,
                                                        token_x_input,
                                                        token_y_input,
                                                        frame_mask_src,
                                                        frame_mask_tag,
                                                        token_mask,
                                                        initial_pos,
                                                        sampling_temperature,
                                                        end_iterations)
        else:
            _, _, _, _, _, frame_out, token_out = build(params,
                                                        frame_input,
                                                        cat_mask_src,
                                                        cat_mask_tag,
                                                        token_x_input,
                                                        token_y_input,
                                                        frame_mask_src,
                                                        frame_mask_tag,
                                                        token_mask)

        if params.use_language:
            token_out = anonymize(token_out)
        if params.use_video:
            frame_out = anonymize(frame_out)

        return token_out, frame_out

    return infrence_model
