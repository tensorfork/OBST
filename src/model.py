"""
Contains all necessary functions to build a model graph
TODO(Lucas): Write docstrings for all functions
"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .dataclass import BlockConfig, ModelParameter
from .optimizers import import_float
from .utils_core import default
from .utils_mtf import activate, anonymize, anonymize_dim, concat, deduplicate, random_name, slice

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))


def _get_attention_dim(params: ModelParameter, block_input: typing.Union[mtf.Tensor, mtf.Shape]) -> ATTENTION_DIM:
    if isinstance(block_input, mtf.Tensor):
        block_input = block_input.shape
    attention_dims = (block_input - params.feature_dims - params.intermediate)[1:]  # Ex: Shape[Sequence, Width, Height]
    idx = params.attention_idx % len(attention_dims)
    dim = attention_dims[idx]
    return ATTENTION_DIM(idx, dim)


def _get_variable(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                  initializer: typing.Callable) -> mtf.Tensor:
    return mtf.get_variable(params.mesh, random_name(), deduplicate(shape), dtype=params.variable_dtype,
                            initializer=initializer)


def _orthogonal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape]) -> mtf.Tensor:
    return _get_variable(params, shape, tf.random_normal_initializer(stddev=0.02))


def _normal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                stddev: float = 0.02, mean: float = 0.) -> mtf.Tensor:
    return _get_variable(params, shape, tf.random_normal_initializer(stddev=stddev, mean=mean))


def _linear(params: ModelParameter, block_input: mtf.Tensor, old: typing.List[mtf.Dimension],
            new: typing.List[mtf.Dimension]) -> mtf.Tensor:
    with tf.variable_scope(random_name()):
        return mtf.einsum([block_input, _orthogonal_var(params, old + new)],
                          deduplicate((block_input.shape - old).dims + new))


def _linear_to_features(params: ModelParameter, block_input: mtf.Tensor,
                        old: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, default(old, params.feature_dims), params.feature_dims)


def _linear_from_features(params: ModelParameter, block_input: mtf.Tensor,
                          new: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, params.feature_dims, default(new, params.intermediate))


def _communicating_linear(params: ModelParameter, block_input: mtf.Tensor):
    return _linear_to_features(params, block_input, params.intermediate)


def _embed(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
           name_extras: typing.Tuple[str]) -> mtf.Tensor:
    params.embedding_param_count = params.embedding_param_count + np.prod([s.size for s in shape])
    return _normal_var(params, shape, params.embedding_stddev)


def _all_mean(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple):
    # maybe use einsum instead of mean
    return (mtf.one_hot(mtf.import_fully_replicated(params.mesh,
                                                    import_float(params.attention_idx), [], str(params.attention_idx)),
                        params.head_dim)
            * mtf.reduce_mean(block_input, reduced_dim=params.head_dim))


def _attention(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]):
    idx, dim = _get_attention_dim(params, block_input)
    params.attention_idx += 1
    tmp = anonymize_dim(dim)
    base = activate(_linear_from_features(params, block_input))

    key = bias = 0
    if 'embedded' in name_extras or 'context' in name_extras:
        key = _communicating_linear(params, base) * dim.size ** -0.5
    if 'embedded' in name_extras or 'positional' in name_extras:
        bias = _embed(params, [dim] + params.feature_dims, tuple())
    key = anonymize(key + bias, dim)
    val = _communicating_linear(params, base)
    qry = _communicating_linear(params, base)

    if 'linear' in name_extras:
        return mtf.einsum([mtf.softplus(qry),
                           anonymize(key, [params.key_dim] + [dim] * (idx in params.masked_attention_dimensions)),
                           anonymize(mtf.softplus(val), params.key_dim)]
                          + ([mtf.cast(mtf.less(mtf.range(params.mesh, dim, dtype=tf.int32),
                                                mtf.range(params.mesh, tmp, dtype=tf.int32)),
                                       params.variable_dtype.activation_dtype)
                              ] if idx in params.masked_attention_dimensions else []),
                          output_shape=block_input.shape)

    lgt = mtf.einsum([qry, key], reduced_dims=[params.key_dim])

    if idx in params.masked_attention_dimensions:  # it's auto-regressive
        lgt += mtf.cast(mtf.less(mtf.range(params.mesh, dim, tf.int32),
                                 mtf.range(params.mesh, tmp, tf.int32)),
                        params.variable_dtype.activation_dtype) * -1e12

    lgt = mtf.exp(lgt - mtf.reduce_max(mtf.stop_gradient(lgt), reduced_dim=tmp))
    return mtf.einsum([lgt, anonymize(val, dim)], block_input.shape) / mtf.reduce_sum(lgt, reduced_dim=tmp)


def _rezero(params, block_input: mtf.Tensor, name_extras: typing.Tuple[str]) -> mtf.Tensor:
    with tf.variable_scope(random_name()):
        return block_input * _get_variable(params, [], tf.constant_initializer(0))


def _feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]) -> mtf.Tensor:
    if 'group' in name_extras:
        intermediate = [params.head_dim,
                        anonymize_dim(params.key_dim, params.key_dim.size * params.group_linear_factor)]
    else:
        intermediate = params.intermediate
    return _linear_to_features(params, activate(_linear_from_features(params, block_input, intermediate)), intermediate)


def _norm(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]) -> mtf.Tensor:
    normalized_shape = block_input.shape - [params.key_dim]
    if 'instance' not in name_extras:
        normalized_shape = normalized_shape - [_get_attention_dim(params, block_input).dim]
    if 'group' not in name_extras:
        normalized_shape = normalized_shape - [params.head_dim]

    block_input -= mtf.reduce_mean(block_input, output_shape=normalized_shape)
    block_input *= mtf.rsqrt(1e-6 + mtf.reduce_mean(mtf.square(block_input), output_shape=normalized_shape))
    block_input *= _normal_var(params, params.feature_dims, mean=1)
    block_input += _normal_var(params, params.feature_dims, mean=0)
    return block_input


def _activate(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]):
    return activate(block_input)


def _convolution(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]):
    convolution_size = 16
    if len(name_extras) == 0:
        convolution_size = int(name_extras[0])
    idx, dim = _get_attention_dim(params, block_input)
    anonymous_block_input = anonymize(block_input, dim)
    indexed = mtf.Dimension("indexed", convolution_size)
    one_hot = mtf.range(params.mesh, indexed, params.variable_dtype.activation_dtype)
    one_hot -= params.convolution_size
    one_hot += mtf.range(params.mesh, dim, params.variable_dtype.activation_dtype)
    one_hot = mtf.one_hot(one_hot, dim)
    output = mtf.einsum([one_hot, anonymous_block_input], block_input.shape + [indexed])
    output = _linear(params, output, [indexed] + params.feature_dims, params.intermediate)
    output = activate(output)
    return _communicating_linear(params, output)


LAYER_FUNCTIONS = {'feed_forward': _feed_forward,
                   'attention':    _attention,
                   'norm':         _norm,
                   'rezero':       _rezero,
                   'embed':        _embed,
                   'all_mean':     _all_mean,
                   'activation':   _activate,
                   'convolution':  _convolution
                   }


def _block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor) -> mtf.Tensor:
    out = block_input
    for layer in block_part_config.layer:
        name, *extras = layer.split('-')
        out = LAYER_FUNCTIONS[name](params, out, extras)
    if not params.use_revnet and block_part_config.skip:
        out += block_input
    return out


def build(params: ModelParameter,
          vid: typing.Optional[mtf.Tensor],
          txt_src: typing.Optional[mtf.Tensor],
          txt_tgt: typing.Optional[mtf.Tensor],
          vid_msk_src: typing.Optional[mtf.Tensor],
          vid_msk_tag: typing.Optional[mtf.Tensor],
          txt_msk: typing.Optional[mtf.Tensor],
          ) -> typing.Tuple[mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor]:
    """
    Build Mesh Tensorflow graph of a model given parameters previously inserted.
    The model slices the video input itself (to save on TPU CPU <--> TPU Core bandwidth), but needs both
    text source and text target.
    :param params: Instance of ModelParameter for which to build the graph
    :param vid: Optional Video to attend over, length=(context+1)
    :param txt_src: Optional tokenized text source, will be embedded
    :param txt_tgt: Optional tokenized text target, required when source is given
    :param vid_msk_src: Optional mask for zero frames
    :param vid_msk_tag: Optional mask to remove loss for certain video frames
    :param txt_msk: Optional mask to remove loss for certain token positions
    :return: (Generated Video, Total Loss, Video Loss, Token Loss)
    """
    with mtf.utils.outside_all_rewrites(), tf.variable_scope(params.model_mode):
        if txt_msk is None:
            txt_msk = mtf.ones(params.mesh, [], params.variable_dtype.activation_dtype)
        else:
            txt_msk = mtf.cast(txt_msk, params.variable_dtype.activation_dtype)

        if vid_msk_src is None:
            vid_msk_src = mtf.ones(params.mesh, [], params.variable_dtype.activation_dtype)
        else:
            vid_msk_src = mtf.cast(vid_msk_src, params.variable_dtype.activation_dtype)

        if vid_msk_tag is None:
            vid_msk_tag = mtf.ones(params.mesh, [], params.variable_dtype.activation_dtype)
        else:
            vid_msk_tag = mtf.cast(vid_msk_tag, params.variable_dtype.activation_dtype)

        if vid is not None:
            vid = mtf.cast(vid, params.variable_dtype.activation_dtype)

        video_loss: typing.Union[int, mtf.Tensor] = 0
        token_loss: typing.Union[int, mtf.Tensor] = 0
        frame_out: typing.Union[int, mtf.Tensor] = 0
        token_out: typing.Union[int, mtf.Tensor] = 0

        spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

        # Slice and Normalize the Video input add a zero frame memory token.
        if params.use_video:
            context_dimension = vid.shape[1]
            input_features = vid.shape[-1:]
            tgt = slice(vid, 1, context_dimension.size, context_dimension)
            src = slice(vid, 0, context_dimension.size - 1, context_dimension)
            src = src * vid_msk_src + _embed(params, shape=vid.shape[2:], name_extras=tuple()) * (1 - vid_msk_src)
            src = _linear_to_features(params, src, input_features)

        # Language embedding and initial feed forward.
        if params.use_language:
            txt_src = _linear_to_features(params,
                                          mtf.one_hot(txt_src, params.vocab_dim,
                                                      dtype=params.variable_dtype.activation_dtype),
                                          [params.vocab_dim])
            txt_src = _linear(params, txt_src, [txt_tgt.shape[-1], params.key_dim], [params.key_dim])

        # Connect video and language Input.
        if params.use_video and params.use_language:
            src = concat([src, txt_src], spatial_ctx)

        # If language only mode, set the language input as src.
        elif not params.use_video:
            src: mtf.Tensor = txt_src

        if params.use_initial_position_embedding:
            src = src + _embed(params, src.shape[1:-1], name_extras=tuple())

        if params.use_revnet:
            out = (src, None, src, None)

            def _layer_builder(block_input: typing.Tuple[mtf.Tensor], block_config: BlockConfig):
                return mtf.layers.reversible_half_residual_and_swap(*block_input,
                                                                    lambda x: _block_part_fn(params, block_config, x))
        else:
            out = src

            def _layer_builder(block_input: mtf.Tensor, block_config: BlockConfig):
                return mtf.recompute_grad(lambda x: _block_part_fn(params, block_config, x), [block_input])

        for _ in range(params.n_blocks):
            for block_part in params.block_config:
                out = _layer_builder(out, block_part)

        if params.use_revnet:
            out = out[0] + out[2]

        # Language Loss
        if params.use_language:
            token_out = _linear_from_features(params, slice(out, 0, params.language_token_patch, spatial_ctx),
                                              [txt_tgt.shape[-1], params.vocab_dim])
            cross_entropy = mtf.layers.softmax_cross_entropy_with_logits(token_out, txt_tgt, params.vocab_dim,
                                                                         params.z_loss)
            token_loss = mtf.reduce_mean(txt_msk * cross_entropy)

        # Video Loss
        if params.use_video:
            out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)
            frame_out = mtf.sigmoid(_linear_from_features(params, out, input_features))
            video_loss: mtf.Tensor = mtf.reduce_mean(mtf.abs(frame_out - tgt) * vid_msk_tag)

        params.layer_idx = 0

        loss = video_loss + token_loss
        video_loss = video_loss * vid_msk_tag.size / mtf.reduce_sum(vid_msk_tag)
        token_loss = token_loss * txt_msk.size / mtf.reduce_sum(txt_msk)

        return loss, video_loss, token_loss, frame_out, token_out
