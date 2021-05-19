import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .backend import linear, linear_from_features, linear_to_features
from .embedding import embed
from .frontend import block_part_fn
from .revnet import RevGradOp
from ..dataclass import BlockArgs, BlockConfig, ModelParameter
from ..mtf_wrapper import (add_n, cast, constant_scalar, dropout, einsum, exp, log, one_hot, ones, reciprocal,
                           reduce_logsumexp, reduce_max, reduce_sum, sigmoid, sign, zeros_like)
from ..utils_mtf import concat, head_argmax, head_embed, slice, weighted_add

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def _default_ones(params: ModelParameter, inp: typing.Optional[mtf.Tensor]) -> mtf.Tensor:
    if inp is None:
        return ones(params.mesh, [], params.variable_dtype.activation_dtype)
    return cast(inp, params.variable_dtype.activation_dtype)


def build(params: ModelParameter,
          vid: typing.Optional[mtf.Tensor],
          cat_msk_src: typing.Optional[mtf.Tensor],
          cat_msk_tgt: typing.Optional[mtf.Tensor],
          txt_src: typing.Optional[mtf.Tensor],
          txt_tgt: typing.Optional[mtf.Tensor],
          vid_msk_src: typing.Optional[mtf.Tensor],
          vid_msk_tgt: typing.Optional[mtf.Tensor],
          txt_msk: typing.Optional[mtf.Tensor],
          ) -> typing.Tuple[mtf.Tensor, typing.List, mtf.Tensor, typing.Optional[mtf.Tensor],
                            mtf.Tensor, mtf.Tensor, mtf.Tensor]:
    """
    Build Mesh Tensorflow graph of a model given parameters previously inserted.
    The model slices the video input itself (to save on TPU CPU <--> TPU Core bandwidth), but needs both
    text source and text target.
    :param params: Instance of ModelParameter for which to build the graph
    :param vid: Optional Video to attend over, length=(context+1)
    :param cat_msk_src: Optional mask for zero frames
    :param cat_msk_tgt: Optional mask to remove loss for certain video frames
    :param txt_src: Optional tokenized text source, will be embedded
    :param txt_tgt: Optional tokenized text target, required when source is given
    :param vid_msk_src: Optional mask for zero frames
    :param vid_msk_tgt: Optional mask to remove loss for certain video frames
    :param txt_msk: Optional mask to remove loss for certain token positions
    :return: (Generated Video, Total Loss, Video Loss, Token Loss)
    """
    with mtf.utils.outside_all_rewrites(), tf1.variable_scope(params.model_mode):
        cat_msk_src = _default_ones(params, cat_msk_src)
        cat_msk_tgt = _default_ones(params, cat_msk_tgt)
        vid_msk_src = _default_ones(params, vid_msk_src)
        vid_msk_tgt = _default_ones(params, vid_msk_tgt)
        txt_msk = _default_ones(params, txt_msk)
        if vid is not None and not params.use_discrete_video_loss and not params.use_bit_fold_input_pipeline:
            vid = mtf.cast(vid, params.variable_dtype.activation_dtype)

        video_loss: typing.Union[int, mtf.Tensor] = 0
        token_loss: typing.Union[int, mtf.Tensor] = 0
        frame_out: typing.Union[int, mtf.Tensor] = 0
        token_out: typing.Union[int, mtf.Tensor] = 0

        spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

        if params.use_video:
            base_args = BlockArgs(params, vid, [''])
            vid = dropout(vid, rate=params.input_dropout)

            if params.use_bit_fold_input_pipeline:
                vid = mtf.cast(vid, dtype=tf.int64)

                concat_list = []
                for unfold_idx in range(params.fold_count):
                    var = mtf.mod(mtf.floordiv(vid, (2 ** params.bit_fold_value) ** unfold_idx),
                                  (2 ** params.bit_fold_value))
                    var = mtf.cast(var, dtype=tf.uint8)

                    concat_list.append(var)

                vid = mtf.concat(concat_list, 'color_channels')

            if not params.use_discrete_video_loss:
                vid = mtf.cast(vid, params.variable_dtype.activation_dtype) / 255
            context_dimension = vid.shape[1]
            input_features = vid.shape[-1:]
            tgt = slice(vid, 1, context_dimension.size, context_dimension)
            src = slice(vid, 0, context_dimension.size - 1, context_dimension)

            if params.use_discrete_video_loss:
                src = mtf.cast(src, params.variable_dtype.activation_dtype) / (params.color_quantization_value - 1)

                tgt = mtf.reshape(tgt, new_shape=mtf.Shape([params.batch_dim,
                                                            params.sequence_per_head_dim,
                                                            params.head_dim]
                                                           + tgt.shape[2:]))

            if params.empty_frame_embedding is not None:
                embed_args = base_args(params.empty_frame_embedding)
                src = weighted_add(src, embed(embed_args, vid.shape[2:]), vid_msk_src)
                src = weighted_add(src, embed(embed_args, vid.shape[2:]), cat_msk_src)

            src = linear_to_features(base_args(src), input_features)

            for config_idx, config in enumerate(params.input_block_config):
                src = block_part_fn(params, config, src, f'vid_inp{config_idx}')

        # Language embedding and initial feed forward.
        if params.use_language:
            base_args = BlockArgs(params, txt_tgt, [''])
            txt_embd = embed(base_args(params.token_embedding),
                             [params.head_dim, params.vocab_dim] + params.intermediate)
            txt = einsum([txt_embd, *head_embed(params, txt_src)], reduced_dims=[params.vocab_dim, params.head_dim])

            if params.input_dropout > 0:
                txt = dropout(txt, rate=params.input_dropout)

            txt = linear_to_features(base_args(txt), [txt_tgt.shape[-1]] + params.intermediate)

            for config_idx, config in enumerate(params.input_block_config):
                txt = block_part_fn(params, config, txt, f'lang_inp{config_idx}')

        if params.use_video and params.use_language:
            src = concat([src, txt], spatial_ctx)
        elif not params.use_video:
            src: mtf.Tensor = txt

        with tf1.variable_scope('body'):
            if params.use_initial_position_embedding:
                for dim in (src.shape - params.feature_dims).dims[1:]:
                    src += embed(base_args(params.position_embedding), [dim] + params.feature_dims)

            if params.use_revnet:
                out = (src, None, src, None)

                def _layer_builder(block_input: typing.Tuple[mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor],
                                   block_config: BlockConfig, index: int):
                    x1, x1_backwards, x2, x2_backwards = block_input
                    if x1_backwards is None:
                        x1_backwards = zeros_like(x1)
                    if x2_backwards is None:
                        x2_backwards = zeros_like(x2)
                    return RevGradOp(params, block_config, x1, x1_backwards, x2, x2_backwards, str(index)).outputs
            else:
                out = src

                def _layer_builder(block_input: mtf.Tensor, block_config: BlockConfig, index: int):
                    return mtf.recompute_grad(lambda x: block_part_fn(params, block_config, x, str(index)),
                                              [block_input])

            for i in range(params.n_blocks):
                for block_part in params.block_config:
                    out = _layer_builder(out, block_part, i)

            if params.use_revnet:
                out = out[0] + out[2]

        if params.use_language:
            token_out = slice(out, 0, params.language_token_patch, spatial_ctx)

            for config_idx, config in enumerate(params.output_block_config):
                token_out = block_part_fn(params, config, token_out, f'lang_out{config_idx}')

            token_out = linear_from_features(base_args(token_out), [txt_tgt.shape[-1]] + params.vocab_dims)

        if params.use_video:
            frame_out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)

            for config_idx, config in enumerate(params.output_block_config):
                frame_out = block_part_fn(params, config, frame_out, f'vid_out{config_idx}')

            if params.use_discrete_video_loss:

                features_dim = mtf.Dimension("features", frame_out.shape[-1].size * frame_out.shape[-2].size)
                frame_out = mtf.reshape(frame_out, frame_out.shape[:-2] + [features_dim])
                frame_out = mtf.reshape(frame_out,
                                        [params.batch_dim, params.sequence_per_head_dim, params.head_dim]
                                        + frame_out.shape[2:])

                frame_out = linear(base_args(frame_out), [features_dim], [vid.shape[-1], params.discrete_color_dim])

            else:
                frame_out = sigmoid(linear_from_features(base_args(frame_out), vid.shape[-1:]))

        loss_list = []
        accuracy = None

        if params.use_language:
            reduced_shape = token_out.shape - params.vocab_dims
            max_logit = reduce_max(token_out, output_shape=reduced_shape)
            msk = txt_msk * cat_msk_tgt * (1 / txt_tgt.size)
            token_loss = einsum([log(reduce_sum(exp(token_out - max_logit), output_shape=reduced_shape)), msk],
                                output_shape=[])
            token_loss += einsum([token_out, *head_embed(params, txt_tgt), constant_scalar(params, -1), msk],
                                 output_shape=[])
            token_loss += einsum([max_logit, msk], output_shape=[])
            loss_list.append(token_loss)

            if txt_msk is not None:
                token_loss = einsum([constant_scalar(params, txt_msk.size), reciprocal(reduce_sum(txt_msk)),
                                     constant_scalar(params, cat_msk_tgt.size), reciprocal(reduce_sum(cat_msk_tgt)),
                                     token_loss], output_shape=[])

            if params.calc_accuracy:
                accuracy = einsum([cast(mtf.equal(head_argmax(token_out, params.vocab_dims), txt_tgt),
                                        params.variable_dtype.activation_dtype), msk], output_shape=[])

        if params.use_video:

            if params.use_discrete_video_loss:

                mak_per_head_shape = mtf.Shape([params.batch_dim, params.sequence_per_head_dim, params.head_dim])
                _vid_msk_tgt = mtf.reshape(vid_msk_tgt, new_shape=mak_per_head_shape)
                _cat_msk_tgt = mtf.reshape(cat_msk_tgt, new_shape=mak_per_head_shape)

                video_size = constant_scalar(params, 1 / tgt.size)
                video_target = one_hot(tgt, params.discrete_color_dim, dtype=params.variable_dtype.activation_dtype)
                video_loss = einsum([reduce_logsumexp(frame_out, reduced_dim=params.discrete_color_dim), video_size,
                                     _vid_msk_tgt, _cat_msk_tgt], output_shape=[params.head_dim])
                video_loss += einsum([frame_out, video_target, video_size, constant_scalar(params, -1),
                                      _vid_msk_tgt, _cat_msk_tgt], output_shape=[params.head_dim])
                video_loss = reduce_sum(video_loss, output_shape=[])

            else:
                size = constant_scalar(params, 1 / frame_out.size)
                out = frame_out - tgt
                video_loss: mtf.Tensor = einsum([out, vid_msk_tgt, cat_msk_tgt, size, sign(out)], output_shape=[])

            loss_list.append(video_loss)

            if vid_msk_tgt is not None:
                video_loss = einsum([constant_scalar(params, vid_msk_tgt.size), reciprocal(reduce_sum(vid_msk_tgt)),
                                     constant_scalar(params, cat_msk_tgt.size), reciprocal(reduce_sum(cat_msk_tgt)),
                                     video_loss], output_shape=[])
        params.layer_idx = 0

        return add_n(loss_list), loss_list, video_loss, accuracy, token_loss, frame_out, token_out
