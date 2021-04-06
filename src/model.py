"""
Contains all necessary functions to build a model graph
TODO(Lucas): Write docstrings for all functions
"""

import random
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import array_ops, gen_linalg_ops, math_ops, random_ops
from tensorflow.python.ops.init_ops import Initializer

from .dataclass import BlockConfig, ModelParameter
from .utils_core import default
from .utils_mtf import (ACTIVATIONS, OPT_DIMS, SHAPE, activate, add_n, anonymize, anonymize_dim, cast, concat,
                        constant_scalar, deduplicate, dropout, einsum, feature_dims_used, greater_equal, maximum,
                        mtf_range, one_hot, ones, random_name, reciprocal, reduce_logsumexp, reduce_mean, reduce_sum,
                        rsqrt, scoped, shift, sigmoid, sign, slice, zeros_like)

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))


class SoftmaxBackward(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dy: mtf.Tensor, dim: mtf.Dimension, masked: bool):
        super().__init__([x, dy], name=random_name("softmax_backward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.dim = dim
        self.shape: mtf.Shape = x.shape
        self.masked = masked

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        dim_index = self.shape.dims.index(self.dim)
        size = self.dim.size
        dim_index = self.shape.dims.index(self.dim)
        anonymous_dim_index = self.shape.dims.index(anonymize_dim(self.dim))

        def slicewise_fn(x, y):
            if self.masked:
                arange = tf.range(0, self.dim.size)
                msk = tf.reshape(arange, (1, self.dim.size)) < tf.reshape(arange, (self.dim.size, 1))
                msk = tf.cast(msk, x.dtype)
                msk *= 1e12
                shape = [1] * len(self.shape.dims)
                shape[dim_index] = self.dim.size
                shape[anonymous_dim_index] = self.dim.size
                msk = tf.reshape(msk, shape)
                x -= msk
            e = tf.exp(x - tf.reduce_max(x, anonymous_dim_index, True))
            s = tf.reduce_sum(e, anonymous_dim_index)
            r = tf.reciprocal(s)
            dims = ''.join(chr(ord('a') + i) for i in range(len(e.shape)))
            sdims = dims[:anonymous_dim_index] + dims[anonymous_dim_index + 1:]
            reshaped = tf.reshape(s, [1 if i == anonymous_dim_index else k for i, k in enumerate(e.shape)])
            return tf.einsum(f'{dims},{sdims},{dims},{sdims},{dims}->{dims}', e + (1 - size), r, reshaped - e, r, y)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]], lowering.tensors[self.inputs[1]])
        lowering.set_tensor_lowering(self.outputs[0], y)


class Softmax(mtf.Operation):
    def __init__(self, x: mtf.Tensor, dim: mtf.Dimension, masked: bool):
        super().__init__([x], name=random_name("softmax_forward"))
        self._outputs = [mtf.Tensor(self, x.shape, x.dtype)]
        self.dim = dim
        self.shape: mtf.Shape = x.shape
        self.masked = masked

    def gradient(self, grad_ys):
        return SoftmaxBackward(self.inputs[0], grad_ys[0], self.dim, self.masked).outputs

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)
        dim_index = self.shape.dims.index(self.dim)
        anonymous_dim_index = self.shape.dims.index(anonymize_dim(self.dim))

        def slicewise_fn(x):
            if self.masked:
                arange = tf.range(0, self.dim.size)
                msk = tf.reshape(arange, (1, self.dim.size)) < tf.reshape(arange, (self.dim.size, 1))
                msk = tf.cast(msk, x.dtype)
                msk *= 1e12
                shape = [1] * len(self.shape.dims)
                shape[dim_index] = self.dim.size
                shape[anonymous_dim_index] = self.dim.size
                msk = tf.reshape(msk, shape)
                x -= msk
            e = tf.exp(x - tf.reduce_max(x, anonymous_dim_index, True))
            return e / tf.reduce_sum(e, anonymous_dim_index, True)

        y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
        lowering.set_tensor_lowering(self.outputs[0], y)


def _get_attention_dim(params: ModelParameter, block_input: typing.Union[mtf.Tensor, mtf.Shape]) -> ATTENTION_DIM:
    if isinstance(block_input, mtf.Tensor):
        block_input = block_input.shape
    attention_dims = (block_input - params.feature_dims - params.intermediate)[1:]  # Ex: Shape[Sequence, Width, Height]
    idx = params.attention_idx % len(attention_dims)
    dim = attention_dims[idx]
    return ATTENTION_DIM(idx, dim)


def _get_variable(params: ModelParameter, shape: SHAPE, initializer: typing.Callable) -> mtf.Tensor:
    return scoped(random_name("get_variable"), mtf.get_variable, params.mesh, random_name("get_variable"),
                  deduplicate(shape), dtype=params.variable_dtype, initializer=initializer)


class OrthogonalInit(Initializer):
    def __init__(self, params: ModelParameter, shape: SHAPE, fan_in_dims: OPT_DIMS = None):
        if fan_in_dims is None:
            fan_in_dims = []
        self.params = params
        self.sizes = [d.size for d in shape]
        self.seed = random.randint(0, 2 ** 32)
        sizes = [d.size for d in shape - fan_in_dims]
        features_used = feature_dims_used(params, shape)
        if features_used and shape.index(params.key_dim) == len(sizes) - 1:
            fan_in = np.prod(sizes[:-2])
        elif features_used:
            fan_in = np.prod([d.size for d in params.feature_dims])
        elif len(sizes) == 2:
            fan_in = sizes[0]
        else:
            raise ValueError(f"Shape: {shape}\nParams: {params}\nFeaturesUsed: {features_used}")
        fan_in *= np.prod([d.size for d in fan_in_dims])
        fan_out = np.prod(sizes) // fan_in
        self.transpose = transpose = fan_out > fan_in
        self.shape = (fan_out, fan_in) if transpose else (fan_in, fan_out)

    def __call__(self, shape, dtype=None, partition_info=None):
        q, r = gen_linalg_ops.qr(random_ops.random_normal(self.shape, dtype=tf.float32, seed=self.seed))
        q *= math_ops.sign(array_ops.diag_part(r))
        if self.transpose:
            q = array_ops.matrix_transpose(q)
        return tf.cast(array_ops.reshape(q, self.sizes) / self.params.n_blocks ** 0.5, dtype)


def _orthogonal_var(params: ModelParameter, shape: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                    fan_in_dims: OPT_DIMS = None) -> mtf.Tensor:
    shape = deduplicate(shape)
    return scoped("orthogonal_var", _get_variable, params, shape, OrthogonalInit(params, shape, fan_in_dims))


def _normal_var(params: ModelParameter, shape: SHAPE, stddev: float = 0.02, mean: float = 0.) -> mtf.Tensor:
    shape = deduplicate(shape)
    return scoped("normal_var", _get_variable, params, shape, tf.random_normal_initializer(stddev=stddev, mean=mean))


def _linear(params: ModelParameter, block_input: mtf.Tensor, old: typing.List[mtf.Dimension],
            new: typing.List[mtf.Dimension]) -> mtf.Tensor:
    return einsum([block_input, _orthogonal_var(params, old + new)],
                  deduplicate((block_input.shape - old).dims + new))


def _linear_to_features(params: ModelParameter, block_input: mtf.Tensor,
                        old: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, default(old, params.feature_dims), params.feature_dims)


def _linear_from_features(params: ModelParameter, block_input: mtf.Tensor,
                          new: typing.Optional[typing.List[mtf.Dimension]] = None) -> mtf.Tensor:
    return _linear(params, block_input, params.feature_dims, default(new, params.intermediate))


def _communicating_linear(params: ModelParameter, block_input: mtf.Tensor):
    return _linear_to_features(params, block_input, params.intermediate)


def _embed(params: ModelParameter, shape: SHAPE) -> mtf.Tensor:
    return scoped("embed", _normal_var, params, shape, params.embedding_stddev)


def _all_mean(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.Tuple):
    return einsum([block_input,
                   one_hot(constant_scalar(params, _get_attention_dim(params, block_input).index / block_input.size),
                           params.head_dim)],
                  reduced_dims=[params.head_dim])


def compare_range(params: ModelParameter, dim0: mtf.Dimension, dim1: mtf.Dimension, comparison):
    with tf.variable_scope(f"compare{dim0.name}_{dim1.name}"):
        return cast(comparison(mtf_range(params.mesh, dim0, tf.bfloat16),
                               mtf_range(params.mesh, dim1, tf.bfloat16)),
                    params.variable_dtype.activation_dtype)


def _attention(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    idx, dim = _get_attention_dim(params, block_input)
    params.attention_idx += 1
    tmp = anonymize_dim(dim)
    base = activate(name_extras, _linear_from_features(params, block_input))
    linear = 'linear' in name_extras
    masked = idx in params.masked_attention_dimensions

    key = 0
    if 'embedded' in name_extras or 'context' in name_extras:
        key = _communicating_linear(params, base) * dim.size ** -0.5
    if 'embedded' in name_extras or 'positional' in name_extras:
        key += _embed(params, [dim] + params.feature_dims)
    val = _communicating_linear(params, base)
    qry = _communicating_linear(params, base)
    if 'activate_val' in name_extras:
        val = activate(name_extras, val)
    if 'activate_key' in name_extras:
        key = activate(name_extras, key)
    if 'activate_qry' in name_extras:
        qry = activate(name_extras, qry)
    val_dim = params.key_dim if linear else dim
    key = anonymize(key, dim)
    val = anonymize(val, val_dim)
    inputs = [qry, anonymize(key, [params.key_dim] * linear + [dim] * (masked or not linear))]
    if masked and linear:
        inputs.append(compare_range(params, dim, tmp, greater_equal))
    if all(f'kernel_{k}' not in name_extras for k in ['softmax'] + list(ACTIVATIONS.keys())):
        return einsum(inputs + [val], output_shape=block_input.shape)
    lgt = einsum(inputs, reduced_dims=[dim if linear else params.key_dim])
    return einsum(Softmax(lgt, dim, masked).outputs + [val], block_input.shape)


def _rezero(params, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return block_input * _get_variable(params, [], tf.constant_initializer(0))


def _feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    if 'group' in name_extras:
        intermediate = [params.head_dim,
                        anonymize_dim(params.key_dim, params.key_dim.size * params.group_linear_factor)]
    else:
        intermediate = params.intermediate

    def _from_feat():
        return _linear_from_features(params, block_input, intermediate)

    mid = activate(name_extras, _from_feat())
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += activate(name_extras, _from_feat())
    return _linear_to_features(params, mid, intermediate)


def _norm(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    normalized_shape = block_input.shape - [params.key_dim]
    if 'instance' not in name_extras:
        normalized_shape = normalized_shape - [_get_attention_dim(params, block_input).dim]
    if 'group' not in name_extras:
        normalized_shape = normalized_shape - [params.head_dim]
    if 'mean' in name_extras:
        block_input -= reduce_mean(block_input, output_shape=normalized_shape)
    scale = []
    if 'std' in name_extras:
        scale.append(rsqrt(1e-6 + einsum([block_input, block_input,
                                          constant_scalar(params, normalized_shape.size / block_input.size)],
                                         output_shape=normalized_shape)))
    if 'scale' in name_extras:
        scale.append(_normal_var(params, params.feature_dims, mean=1))
    if scale:
        block_input = mtf.einsum([block_input] + scale, output_shape=block_input.shape)
    if 'shift' in name_extras:
        block_input += _normal_var(params, params.feature_dims, mean=0)
    return block_input


def _activate(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    return activate(name_extras, block_input)


def _convolution(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    idx, dim = _get_attention_dim(params, block_input)
    convolution_size = 16
    if len(name_extras) > 0 and name_extras[-1].isdigit():
        convolution_size = int(name_extras[-1])
    if "gather" in name_extras:
        anonymous_block_input = anonymize(block_input, dim)
        indexed = mtf.Dimension("indexed", convolution_size)
        hot = mtf_range(params.mesh, indexed, params.variable_dtype.activation_dtype)
        hot -= params.convolution_size
        hot += mtf_range(params.mesh, dim, params.variable_dtype.activation_dtype)
        hot = maximum(hot, 0)
        hot = one_hot(hot, dim)
        output = einsum([hot, anonymous_block_input], block_input.shape + [indexed])
        output = _linear(params, output, [indexed] + params.feature_dims, params.intermediate)
        output = activate(name_extras, output)
        return _communicating_linear(params, output)
    out = [shift(_linear_from_features(params, block_input), i, dim, False) for i in range(convolution_size)]
    return _communicating_linear(params, activate(name_extras, add_n(out)))


LAYER_FUNCTIONS = {'feed_forward': _feed_forward,
                   'attention':    _attention,
                   'norm':         _norm,
                   'rezero':       _rezero,
                   'embed':        _embed,
                   'all_mean':     _all_mean,
                   'activation':   _activate,
                   'convolution':  _convolution
                   }


def _block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor,
                   name_prefix: str = 'block') -> mtf.Tensor:
    out = block_input
    with tf.variable_scope(random_name(f"{name_prefix}_")):
        for layer in block_part_config.layer:
            name, *extras = layer.split('-')
            out = scoped(name, LAYER_FUNCTIONS[name], params, out, extras)

        if not block_part_config.use_revnet and block_part_config.skip:
            out += block_input

    return out


class RevGradOp(mtf.Operation):
    """Operation to implement custom gradients.

    See comments on custom_gradient() below.
    """

    def __init__(self, params, block_config, x1, x1_backwards, x2, x2_backwards, index):
        graph: mtf.Graph = x1.graph
        prev_ops = len(graph.operations)
        y1 = x1 + _block_part_fn(params, block_config, x2, index)
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
    return cast(default(inp, ones(params.mesh, [], params.variable_dtype.activation_dtype)),
                params.variable_dtype.activation_dtype)


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
    with mtf.utils.outside_all_rewrites(), tf.variable_scope(params.model_mode):
        cat_msk_src = default_ones(params, cat_msk_src)
        cat_msk_tgt = default_ones(params, cat_msk_tgt)
        vid_msk_src = default_ones(params, vid_msk_src)
        vid_msk_tgt = default_ones(params, vid_msk_tgt)
        txt_msk = default_ones(params, txt_msk)
        if vid is not None:
            vid = mtf.cast(vid, params.variable_dtype.activation_dtype)

        video_loss: typing.Union[int, mtf.Tensor] = 0
        token_loss: typing.Union[int, mtf.Tensor] = 0
        frame_out: typing.Union[int, mtf.Tensor] = 0
        token_out: typing.Union[int, mtf.Tensor] = 0

        spatial_ctx: mtf.Dimension = txt_tgt.shape[-2] if params.use_language else vid.shape[2]

        if params.use_video and params.input_dropout > 0:
            vid = dropout(vid, rate=params.input_dropout)
        if params.use_video:
            context_dimension = vid.shape[1]
            input_features = vid.shape[-1:]
            tgt = slice(vid, 1, context_dimension.size, context_dimension)
            src = slice(vid, 0, context_dimension.size - 1, context_dimension)
            src = src * vid_msk_src + _embed(params, shape=vid.shape[2:]) * (1 - vid_msk_src)
            src = src * cat_msk_src + _embed(params, shape=vid.shape[2:]) * (1 - cat_msk_src)

            src = _linear_to_features(params, src, input_features)

            for config_idx, config in enumerate(params.input_block_config):
                src = _block_part_fn(params, config, src, f'vid_inp{config_idx}')

        # Language embedding and initial feed forward.
        if params.use_language:
            txt = einsum([one_hot(txt_src, params.vocab_dim, dtype=params.variable_dtype.activation_dtype),
                          _embed(params, [params.vocab_dim] + params.intermediate)], reduced_dims=[params.vocab_dim])

            if params.input_dropout > 0:
                txt = dropout(txt, rate=params.input_dropout)

            txt = _linear_to_features(params, txt, [txt_tgt.shape[-1]] + params.intermediate)

            for config_idx, config in enumerate(params.input_block_config):
                txt = _block_part_fn(params, config, txt, f'lang_inp{config_idx}')

        if params.use_video and params.use_language:
            src = concat([src, txt], spatial_ctx)
        elif not params.use_video:
            src: mtf.Tensor = txt

        with tf.variable_scope('body'):
            if params.use_initial_position_embedding:
                for dim in (src.shape - params.feature_dims).dims[1:]:
                    src += _embed(params, [dim] + params.feature_dims)

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
                    return mtf.recompute_grad(lambda x: _block_part_fn(params, block_config, x, str(index)),
                                              [block_input])

            for i in range(params.n_blocks):
                for block_part in params.block_config:
                    out = _layer_builder(out, block_part, i)

            if params.use_revnet:
                out = out[0] + out[2]

        if params.use_language:
            token_out = slice(out, 0, params.language_token_patch, spatial_ctx)

            for config_idx, config in enumerate(params.output_block_config):
                token_out = _block_part_fn(params, config, token_out, f'lang_out{config_idx}')

            token_out = _linear_from_features(params, token_out, [txt_tgt.shape[-1], params.vocab_dim])

        if params.use_video:
            frame_out = slice(out, params.language_token_patch * params.use_language, out.shape[2].size, spatial_ctx)

            for config_idx, config in enumerate(params.output_block_config):
                frame_out = _block_part_fn(params, config, frame_out, f'vid_out{config_idx}')

            frame_out = sigmoid(_linear_from_features(params, frame_out, vid.shape[-1:]))

        loss_list = []
        accuracy = None

        if params.use_language:
            size = constant_scalar(params, 1 / txt_tgt.size)
            target = one_hot(txt_tgt, params.vocab_dim, dtype=params.variable_dtype.activation_dtype)
            token_loss = einsum([reduce_logsumexp(token_out, reduced_dim=params.vocab_dim), size, txt_msk, cat_msk_tgt],
                                output_shape=[])
            token_loss += einsum([token_out, target, size, constant_scalar(params, -1), txt_msk, cat_msk_tgt],
                                 output_shape=[])
            loss_list.append(token_loss)

            if txt_msk is not None:
                token_loss = einsum([constant_scalar(params, txt_msk.size), reciprocal(reduce_sum(txt_msk)),
                                     constant_scalar(params, cat_msk_tgt.size), reciprocal(reduce_sum(cat_msk_tgt)),
                                     mtf.stop_gradient(token_loss)], output_shape=[])

            if params.calc_accuracy:
                accuracy = einsum([cast(mtf.equal(mtf.argmax(mtf.stop_gradient(token_out), params.vocab_dim), txt_tgt),
                                        tf.float32), txt_msk, cat_msk_tgt, size], output_shape=[])

        if params.use_video:
            size = constant_scalar(params, 1 / frame_out.size)
            out = frame_out - tgt
            video_loss: mtf.Tensor = einsum([out, vid_msk_tgt, cat_msk_tgt, size, sign(out)], output_shape=[])
            loss_list.append(video_loss)

            if vid_msk_tgt is not None:
                video_loss = einsum([constant_scalar(params, vid_msk_tgt.size), reciprocal(reduce_sum(vid_msk_tgt)),
                                     constant_scalar(params, cat_msk_tgt.size), reciprocal(reduce_sum(cat_msk_tgt)),
                                     mtf.stop_gradient(video_loss)], output_shape=[])
        params.layer_idx = 0

        video_loss = video_loss * vid_msk_tgt.size / reduce_sum(vid_msk_tgt)
        token_loss = token_loss * txt_msk.size / reduce_sum(txt_msk)

        return add_n(loss_list), loss_list, video_loss, accuracy, token_loss, frame_out, token_out
