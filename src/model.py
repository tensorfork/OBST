"""
Contains all necessary functions to build a model graph
TODO(Lucas): Write docstrings for all functions
"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf

from .dataclass import BlockConfig, ModelParameter
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
                                                    tf.constant(params.attention_idx, dtype=tf.float32, shape=[]), [],
                                                    str(params.attention_idx)),
                        params.head_dim)
            * mtf.reduce_mean(block_input, reduced_dim=params.head_dim))


def compare_range(params: ModelParameter, dim0: mtf.Dimension, dim1: mtf.Dimension, comparison):
    return mtf.cast(comparison(mtf.range(params.mesh, dim0, tf.bfloat16), mtf.range(params.mesh, dim1, tf.bfloat16)),
                    params.variable_dtype.activation_dtype)


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
                          + ([compare_range(params, dim, tmp, mtf.less)] if
                             idx in params.masked_attention_dimensions else []),
                          output_shape=block_input.shape)

    lgt = mtf.einsum([qry, key], reduced_dims=[params.key_dim])

    if idx in params.masked_attention_dimensions:  # it's auto-regressive
        lgt += compare_range(params, dim, tmp, mtf.less) * -1e12

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
    if 'mean' in name_extras:
        block_input -= mtf.reduce_mean(block_input, output_shape=normalized_shape)
    if 'std' in name_extras:
        block_input *= mtf.rsqrt(1e-6 + mtf.reduce_mean(mtf.square(block_input), output_shape=normalized_shape))
    if 'scale' in name_extras:
        block_input *= _normal_var(params, params.feature_dims, mean=1)
    if 'shift' in name_extras:
        block_input += _normal_var(params, params.feature_dims, mean=0)
    return block_input


def _activate(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]):
    return activate(block_input)


def _convolution(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple[str]):
    idx, dim = _get_attention_dim(params, block_input)
    convolution_size = 16
    if len(name_extras) > 0 and name_extras[-1].isdigit():
        convolution_size = int(name_extras[-1])
    if "gather" in name_extras:
        anonymous_block_input = anonymize(block_input, dim)
        indexed = mtf.Dimension("indexed", convolution_size)
        one_hot = mtf.range(params.mesh, indexed, params.variable_dtype.activation_dtype)
        one_hot -= params.convolution_size
        one_hot += mtf.range(params.mesh, dim, params.variable_dtype.activation_dtype)
        one_hot = mtf.maximum(one_hot, 0)
        one_hot = mtf.one_hot(one_hot, dim)
        output = mtf.einsum([one_hot, anonymous_block_input], block_input.shape + [indexed])
        output = _linear(params, output, [indexed] + params.feature_dims, params.intermediate)
        output = activate(output)
        return _communicating_linear(params, output)
    out = [mtf.shift(_linear_from_features(params, block_input), i, dim, False) for i in range(convolution_size)]
    return _communicating_linear(params, activate(mtf.add_n(out)))


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


class RevGradOp(mtf.Operation):
    """Operation to implement custom gradients.

    See comments on custom_gradient() below.
    """

    def __init__(self, params, block_config, x1, x1_backwards, x2, x2_backwards):
        graph: mtf.Graph = x1.graph
        prev_ops = len(graph.operations)
        y1 = x1 + _block_part_fn(params, block_config, x2)
        fn_outputs = [x2, x2_backwards, y1, x1_backwards]
        forward_operations = graph.operations[prev_ops:]
        new_outputs = set()
        new_inputs = set()
        for op in forward_operations:
            new_inputs.update(set(op.inputs))
            if not isinstance(op, mtf.Variable):
                new_outputs.update(set(op.outputs))
        explicit_inputs = [x1, x1_backwards, x2, x2_backwards]
        variables = [t for t in list(new_inputs - new_outputs - set(explicit_inputs)) if t.dtype.is_floating]
        super(RevGradOp, self).__init__(explicit_inputs + variables + fn_outputs, x1.mesh,
                                        random_name("custom_gradient"))
        # Make sure no one uses the internals of this function, since the gradients
        #  will probably not work correctly.
        for t in new_outputs - set(fn_outputs):
            t.usable = False

        self._graph: mtf.Graph = x1.graph
        self._x2: mtf.Tensor = x2
        self._y1: mtf.Tensor = y1
        self._variables: typing.List[mtf.Variable] = variables
        self._fn_outputs: typing.List[mtf.Tensor] = fn_outputs
        self._outputs: typing.List[mtf.Tensor] = [mtf.Tensor(self, x.shape, x.dtype, index=i)
                                                  for i, x in enumerate(fn_outputs)]
        self._forward_operations = forward_operations[:-1]

    def lower(self, lowering):
        for fn_output, output in zip(self._fn_outputs, self._outputs):
            lowering.set_tensor_lowering(output, lowering.tensors[fn_output])

    def gradient(self, grad_ys, params: typing.Optional[typing.List[mtf.Operation]] = None):
        dy2, dy2_backwards, dy1, dy1_backwards = grad_ys
        x2 = self._x2 if dy2_backwards is None else dy2_backwards
        f_again_ops, mapping = self._graph.clone_operations(self._forward_operations, {self._x2: x2})
        fx2 = mapping[self._forward_operations[-1].outputs[0]]
        # figure out what Tensors are downstream of xs
        downstream = set([x2] + self._variables)
        for op in f_again_ops:
            if op.has_gradient and set(op.inputs) & downstream:
                downstream |= set(op.outputs)
        tensor_to_gradient = {fx2: dy1}
        if params is None:
            yield dy1
            yield (self._y1 if dy1_backwards is None else dy1_backwards) - fx2
            with tf.variable_scope(fx2.graph.captured_variable_scope):
                for op in f_again_ops[::-1]:
                    grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
                    if not op.has_gradient or not any(grad_outputs) or not set(op.inputs) & downstream:
                        continue
                    with tf.variable_scope(op.name + "/revnet/gradients"):
                        for inp, grad in zip(op.inputs, op.gradient(grad_outputs)):
                            if inp not in downstream or grad is None:
                                continue
                            if inp in tensor_to_gradient:
                                tensor_to_gradient[inp] += grad
                            else:
                                tensor_to_gradient[inp] = grad
            yield dy2 + tensor_to_gradient[x2]
            yield x2
            for g in (tensor_to_gradient.get(x, None) for x in self._variables):
                yield g
            return
        tensor_to_gradient = {fx2: [0, 0, dy1]}
        yield params[0], dy1
        yield params[1], (self._y1 if dy1_backwards is None else dy1_backwards) - fx2
        yield params[3], x2
        with tf.variable_scope(fx2.graph.captured_variable_scope):
            for op in f_again_ops[::-1]:
                grad_outputs = []
                for out in op.outputs:
                    grad = tensor_to_gradient.get(out)
                    if grad is None:
                        grad_outputs.append(None)
                        continue
                    grad_outputs.append(grad[2])
                    grad[0] += 1
                    if grad[0] == len(grad[2].operation.inputs):
                        del tensor_to_gradient[out]
                if not op.has_gradient or not any(grad_outputs) or not set(op.inputs) & downstream:
                    continue
                for inp, grad in zip(op.inputs, op.gradient(grad_outputs)):
                    if inp not in downstream or grad is None:
                        continue
                    if inp in tensor_to_gradient:
                        grad_list = tensor_to_gradient[inp]
                        grad_list[1] += 1
                        with tf.variable_scope(op.name + "/revnet/gradients"):
                            grad_list[2] += grad
                    else:
                        tensor_to_gradient[inp] = grad_list = [0, 1, grad]
                    if len(inp.operation.outputs) != grad_list[1]:
                        continue
                    if inp not in self._variables:
                        continue
                    yield params[4 + self._variables.index(inp)], grad_list[2]
        yield params[2], dy2 + tensor_to_gradient[x2][2]


def default_ones(params, inp):
    return mtf.cast(default(inp, mtf.ones(params.mesh, [], params.variable_dtype.activation_dtype)),
                    params.variable_dtype.activation_dtype)


def build(params: ModelParameter,
          vid: typing.Optional[mtf.Tensor],
          cat_mask_src: typing.Optional[mtf.Tensor],
          cat_mask_tgt: typing.Optional[mtf.Tensor],
          txt_src: typing.Optional[mtf.Tensor],
          txt_tgt: typing.Optional[mtf.Tensor],
          vid_msk_src: typing.Optional[mtf.Tensor],
          vid_msk_tgt: typing.Optional[mtf.Tensor],
          txt_msk: typing.Optional[mtf.Tensor],
          ) -> typing.Tuple[mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor]:
    """
    Build Mesh Tensorflow graph of a model given parameters previously inserted.
    The model slices the video input itself (to save on TPU CPU <--> TPU Core bandwidth), but needs both
    text source and text target.
    :param params: Instance of ModelParameter for which to build the graph
    :param vid: Optional Video to attend over, length=(context+1)
    :param cat_mask_src: Optional mask for zero frames
    :param cat_mask_tgt: Optional mask to remove loss for certain video frames
    :param txt_src: Optional tokenized text source, will be embedded
    :param txt_tgt: Optional tokenized text target, required when source is given
    :param vid_msk_src: Optional mask for zero frames
    :param vid_msk_tgt: Optional mask to remove loss for certain video frames
    :param txt_msk: Optional mask to remove loss for certain token positions
    :return: (Generated Video, Total Loss, Video Loss, Token Loss)
    """
    with mtf.utils.outside_all_rewrites(), tf.variable_scope(params.model_mode):
        cat_mask_src = default_ones(params, cat_mask_src)
        cat_mask_tgt = default_ones(params, cat_mask_tgt)
        txt_msk = default_ones(params, txt_msk)
        vid_msk_src = default_ones(params, vid_msk_src)
        vid_msk_tgt = default_ones(params, vid_msk_tgt)

        if vid is not None:
            vid = mtf.cast(vid, params.variable_dtype.activation_dtype)

        video_loss: typing.Union[int, mtf.Tensor] = 0
        token_loss: typing.Union[int, mtf.Tensor] = 0
        frame_out: typing.Union[int, mtf.Tensor] = 0
        token_out: typing.Union[int, mtf.Tensor] = 0

        spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

        if params.use_video and params.input_dropout > 0:
            vid = mtf.dropout(vid, rate=params.input_dropout)
        if params.use_video:
            context_dimension = vid.shape[1]
            input_features = vid.shape[-1:]
            tgt = slice(vid, 1, context_dimension.size, context_dimension)
            src = slice(vid, 0, context_dimension.size - 1, context_dimension)
            src = src * vid_msk_src + _embed(params, shape=vid.shape[2:], name_extras=tuple()) * (1 - vid_msk_src)
            src = src * cat_mask_src + _embed(params, shape=vid.shape[2:], name_extras=tuple()) * (1 - cat_mask_src)
            src = _linear_to_features(params, src, input_features)

        # Language embedding and initial feed forward.
        if params.use_language:
            txt_src = _linear_to_features(params,
                                          mtf.one_hot(txt_src, params.vocab_dim,
                                                      dtype=params.variable_dtype.activation_dtype),
                                          [params.vocab_dim])
        if params.use_language and params.input_dropout > 0:
            txt_src = mtf.dropout(txt_src, rate=params.input_dropout)
        if params.use_language:
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

            def _layer_builder(block_input: typing.Tuple[mtf.Tensor, mtf.Tensor, mtf.Tensor, mtf.Tensor],
                               block_config: BlockConfig):
                x1, x1_backwards, x2, x2_backwards = block_input
                if x1_backwards is None:
                    x1_backwards = mtf.zeros_like(x1)
                if x2_backwards is None:
                    x2_backwards = mtf.zeros_like(x2)
                return RevGradOp(params, block_config, x1, x1_backwards, x2, x2_backwards).outputs
        else:
            out = src

            def _layer_builder(block_input: mtf.Tensor, block_config: BlockConfig):
                return mtf.recompute_grad(lambda x: _block_part_fn(params, block_config, x), [block_input])

        for _ in range(params.n_blocks):
            for block_part in params.block_config:
                out = _layer_builder(out, block_part)

        if params.use_revnet:
            out = out[0] + out[2]

        if params.use_language:
            token_out = _linear_from_features(params, slice(out, 0, params.language_token_patch, spatial_ctx),
                                              [txt_tgt.shape[-1], params.vocab_dim])
        if params.use_video:
            out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)
            frame_out = mtf.sigmoid(_linear_from_features(params, out, input_features))
        if params.contrastive:
            mask = compare_range(params, params.batch_dim, anonymize_dim(params.batch_dim), mtf.equal) * 2 - 1
        if params.use_language and not params.contrastive:
            targets = mtf.one_hot(txt_tgt, params.vocab_dim, dtype=params.variable_dtype.activation_dtype)
            log_softmax = token_out - mtf.reduce_logsumexp(token_out, params.vocab_dim)
            token_loss = mtf.negative(mtf.reduce_sum(log_softmax * targets)) / txt_tgt.size
        if params.use_language and params.contrastive:
            token_loss = mtf.einsum([token_out, anonymize(token_out, params.batch_dim), mask], output_shape=[])
            token_loss /= token_out.size * params.train_batch_size
        if params.use_video and not params.contrastive:
            video_loss: mtf.Tensor = mtf.reduce_mean(mtf.abs(frame_out - tgt) * vid_msk_tgt * cat_mask_tgt)
        if params.use_video and params.contrastive:
            video_loss = mtf.einsum([frame_out, anonymize(frame_out, params.batch_dim), mask], output_shape=[])
            video_loss /= frame_out.size * params.train_batch_size

        params.layer_idx = 0

        loss = video_loss + token_loss
        video_loss = video_loss * vid_msk_tgt.size / mtf.reduce_sum(vid_msk_tgt)
        token_loss = token_loss * txt_msk.size / mtf.reduce_sum(txt_msk)

        return loss, video_loss, token_loss, frame_out, token_out
