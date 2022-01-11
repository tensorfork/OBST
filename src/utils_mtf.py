import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer

from .dataclass import BlockArgs, ModelParameter
from .mtf_wrapper import cast, mtf_range, reshape, concat as mtf_concat, pad as mtf_pad, mtf_slice, add, multiply, \
    negative
from .utils_core import default, random_name

tf1 = tf.compat.v1

DIM = typing.Union[mtf.Dimension, str]
DIM_LIST = typing.List[mtf.Dimension]
SHAPE = typing.Union[mtf.Shape, DIM_LIST]
TENSORS = typing.List[mtf.Tensor]
OPT_SHAPE = typing.Optional[SHAPE]
OPT_DIMS = typing.Optional[DIM_LIST]
ALL_SHAPES = typing.Union[SHAPE, mtf.Tensor, mtf.Variable]
ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))
LINEAR_SHAPES = typing.NamedTuple("LinearShapes", (('old', DIM_LIST), ('new', DIM_LIST)))


def new_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None,
            new_name: typing.Optional[str] = None):
    """
    Create new mesh tensorflow dimension with optional new size and/or new name to replace the old values with.
    :param dim: Dimension or name of dimension
    :param new_size: Optional new size of mtf dimension
    :param new_name: Optinal new name of dimension
    :return: new mtf.Dimension
    """
    name = default(new_name, dim_name(dim))
    if isinstance(dim, mtf.Dimension):
        return mtf.Dimension(name, default(new_size, dim.size))
    if new_size is None:
        return name
    return mtf.Dimension(name, new_size)


def unanonymize_dim(dim: typing.Union[mtf.Dimension, str], new_size: typing.Optional[int] = None):
    """
    Unanonymize mtf.Dimension by removing a leading underscore, if it exists. Optionally, the size can be changed at
    the same time.
    :param dim: mtf.Dimension to unanonymize
    :param new_size: Optional new size
    :return: mtf.Dimension without leading underscore in name
    """
    name = dim_name(dim)
    if name.startswith('_'):
        name = name[1:]
    return new_dim(dim, new_size, name)


def unanonymize(inp: mtf.Tensor,
                dim: typing.Union[typing.List[typing.Union[mtf.Dimension, str]], typing.Union[mtf.Dimension, str]]
                ) -> mtf.Tensor:
    """
    Remove underscore of the name of a dimension of a tensor. This dereplicates a given dimension from all devices.
    :param inp: tensor to replicate
    :param dim: dimension(s) to replicate
    :return: replicated tensor
    """
    if not isinstance(dim, list):
        dim = [dim]
    shape = inp.shape.dims.copy()
    for cdim in dim:
        cdim = anonymize_dim(dim_name(cdim))
        if not check_for_dim(inp, cdim):
            continue
        shape = [unanonymize_dim(d) if cdim == d.name else d for d in shape]
    if shape != inp.shape.dims:
        if isinstance(dim, mtf.Dimension):
            name = dim.name
        else:
            name = '-'.join(dim_name(d) for d in dim)
        with tf1.variable_scope(f"unanonymize_{name}"):
            return reshape(inp, shape)
    return inp


def anonymize_dim(dim: DIM, new_size: typing.Optional[int] = None):
    """
    Anonymize mtf.Dimension by adding a leading underscore, if it does not exist. Optionally, the size can be changed at
    the same time.
    :param dim: mtf.Dimension to anonymize
    :param new_size: Optional new size
    :return: mtf.Dimension with leading underscore in name
    """
    name = dim_name(dim)
    if not name.startswith('_'):
        name = '_' + name
    return new_dim(dim, new_size, name)


def unbind(tensor: mtf.Tensor, dim: DIM) -> typing.List[mtf.Tensor]:
    if isinstance(dim, mtf.Dimension):
        dim = dim.name
    return [mtf.slice(tensor, i, 1, dim) for i in range(get_dim(tensor, dim).size)]


def squeeze(tensor: mtf.Tensor, dims: typing.Union[SHAPE, DIM]):
    shape = tensor.shape
    if isinstance(dims, (mtf.Dimension, str)):
        dims = [dims]
    dims = dims_from_shape(dims)
    for d in dims:
        shape = shape - [get_dim(shape, d)]
    return mtf.reshape(tensor, shape)


def get_dim(shape: typing.Union[mtf.Tensor, mtf.Shape, typing.List[mtf.Dimension]],
            dim: DIM,
            index=False) -> typing.Union[int, mtf.Dimension]:
    """
    Attempts to get a dimension of a tensor. Raises a ValueError if the dimension does not exist.
    :param shape: shape, tensor or list of dimensions to check in
    :param dim: dimension (or name) to check for
    :param index: whether to return the dimension or its index
    :return: index or dimension
    """
    name = dim_name(dim)
    for idx, cdim in enumerate(shape.shape if isinstance(shape, mtf.Tensor) else shape):
        if cdim.name == name:
            return idx if index else cdim
    raise ValueError(f"Dim {dim} with name {name} not found in shape {shape}")


def concat(tensor_list: typing.List[mtf.Tensor], dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    """
    Concatenate across a given (potentially non-anonymous) dimension in mtf.Tensor. This first anonymizes the dimension
    to concat in the first place, next it concats across the dimension and only then it replicates it on all devices
    again.
    Non-Anonymous shapes are not necessary, as the anonymization can skip itself if it isn't necessary.
    :param tensor_list: mtf.Tensor's to concatenate
    :param dim: dimension or name to concatenate in
    :return: concated tensorlist
    """
    dim = dim_name(dim)
    return unanonymize(mtf_concat([anonymize(t, dim) for t in tensor_list], anonymize_dim(dim)), dim)


def pad(tensor: mtf.Tensor, dim: typing.Union[mtf.Dimension, str], padding: typing.Tuple[int, int]
        ) -> mtf.Tensor:
    """
    Pad across a given (potentially non-anonymous) dimension in mtf.Tensor. This first anonymizes the dimension
    to concat in the first place, next it concats across the dimension and only then it replicates it on all devices
    again.
    Non-Anonymous shapes are not necessary, as the anonymization can skip itself if it isn't necessary.
    :param tensor: mtf.Tensor's to pad
    :param dim: dimension or name to pad in
    :param padding: padding of dimension
    :return: concated tensorlist
    """
    dim = dim_name(dim)
    return mtf_pad(anonymize(tensor, dim), padding, anonymize_dim(dim))


def to_fp32(tensor: mtf.Tensor) -> mtf.Tensor:
    """
    Cast a tensor to float
    :param tensor: tensor to be casted
    :return: casted tensor
    """
    return cast(tensor, tf.float32)


def dim_name(dim: typing.Union[mtf.Dimension, str]) -> str:
    """
    :param dim: Mesh TensorFlow dimension or name of dimension
    :return: name of dimension
    """
    return dim.name if isinstance(dim, mtf.Dimension) else dim


def check_for_dim(inp: typing.Union[typing.List[mtf.Dimension], mtf.Shape, mtf.Tensor],
                  dim: typing.Union[mtf.Dimension, str]) -> bool:
    """
    Check if a dimension exists in a Mesh TensorFlow tensor, shape or list of dimensions
    :param inp: input to check in
    :param dim: dimension to check for
    :return: true if dimension is found
    """
    return any(dim_name(dim) == cdim.name for cdim in (inp.shape if isinstance(inp, mtf.Tensor) else inp))


def deduplicate(inp: SHAPE) -> SHAPE:
    """
    Remove duplicates from any iterable while retaining the order of elements.
    :param inp: iterable to deduplicate
    :return: new, unique iterable of same type as input
    """
    return type(inp)(dict.fromkeys(list(inp)))


def gradient_iterator(params: ModelParameter, op: mtf.Operation, grad_outputs: typing.List[mtf.Tensor]
                      ) -> typing.Iterable[typing.Tuple[mtf.Operation, mtf.Tensor, mtf.Tensor]]:
    from .model.momentumnet import MomentumOperation
    from .model.revnet import RevGradOp
    if isinstance(op, (RevGradOp, MomentumOperation)):
        return op.gradient(grad_outputs, params=op.inputs)
    return zip((op,) * len(op.inputs), op.inputs, op.gradient(grad_outputs))


def anonymize(inp: mtf.Tensor,
              dim: typing.Union[typing.List[typing.Union[mtf.Dimension, str]], typing.Union[mtf.Dimension, str]]
              ) -> mtf.Tensor:
    """
    Add an underscore to the name of a dimension of a tensor. This replicates a given dimension of a tensor on all
    devices.
    :param inp: tensor to replicate
    :param dim: dimension(s) to replicate
    :return: replicated tensor
    """
    if not isinstance(dim, list):
        dim = [dim]
    shape = inp.shape.dims.copy()
    for cdim in dim:
        cdim = unanonymize_dim(dim_name(cdim))
        if not check_for_dim(inp, cdim):
            continue
        shape = [anonymize_dim(d) if cdim == d.name else d for d in shape]
    if shape != inp.shape.dims:
        if isinstance(dim, mtf.Dimension):
            name = dim.name
        else:
            name = '-'.join(dim_name(d) for d in dim)
        with tf1.variable_scope(f"anonymize_{name}"):
            return reshape(inp, shape)
    return inp


class BroadcastForward(mtf.Operation):
    """Broadcast - output dims are a superset of input dims, in any order."""

    def __init__(self, x, output_shape):
        super(BroadcastForward, self).__init__([x], name=random_name("broadcast_forward"))
        self._outputs = [mtf.Tensor(self, output_shape, x.dtype)]
        self._splittable_dims, self._unsplittable_dims = self._initialize_all_dimensions_as_splittable()

    def gradient(self, grad_ys: typing.List[mtf.Tensor]):
        return BroadcastBackward(grad_ys[0], self.inputs[0]).outputs

    def lower(self, lowering: mtf.Lowering):
        inp, out = self.inputs[0], self.outputs[0]
        lowering.tensors[out] = lowering.mesh_impl(self).broadcast_impl(lowering.tensors[inp], inp.shape, out.shape)


class BroadcastBackward(mtf.Operation):
    def __init__(self, grad_y: mtf.Tensor, inp: mtf.Tensor):
        super(BroadcastBackward, self).__init__([grad_y], name=random_name("broadcast_backward"))
        self._outputs = [mtf.Tensor(self, inp.shape, inp.dtype)]
        self._splittable_dims, self._unsplittable_dims = self._initialize_all_dimensions_as_splittable()

    def lower(self, lowering: mtf.Lowering):
        grad, out = self.inputs[0], self.outputs[0]
        dims = [grad.shape.dims.index(d) for d in (grad.shape - out.shape).dims]

        def slicewise_fn(y):
            return tf.reduce_sum(y, dims)

        lowering.tensors[out] = lowering.mesh_impl(self).slicewise(slicewise_fn, lowering.tensors[grad])


def non_replicated_broadcast(x, shape):
    return BroadcastForward(x, mtf.Shape(shape)).outputs[0]


def get_variable(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer, trainable: bool,
                 dtype: mtf.VariableDType):
    full_name = f'{tf1.get_variable_scope().name}/{name}'
    if full_name in params.mesh.graph.name_to_variable:
        return params.mesh.graph.name_to_variable[full_name].outputs[0]
    shape = deduplicate(mtf.Shape(shape))
    var = mtf.Variable(params.mesh, name, shape, dtype, initializer, trainable)
    params.mesh.graph.name_to_variable[full_name] = var
    return var.outputs[0]


def non_replicated_variable(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer, trainable: bool,
                            dtype: mtf.VariableDType):
    var = get_variable(params, name, shape, initializer, trainable, dtype)
    if params.grad_accumulation > 1 and params.batch_splits > 1 and params.split_grad_accumulation:
        return non_replicated_broadcast(var, [params.batch_dim] + dims_from_shape(shape))
    return var


def anonymize_shape(inp: typing.Union[typing.List[mtf.Dimension], mtf.Shape],
                    dim: typing.Union[mtf.Dimension]) -> typing.Union[mtf.Shape, typing.List[mtf.Dimension]]:
    """
    Anonymize one dimension of a given Mesh TensorFlow shape. See anonymize for details on what anonymization does.
    :param inp: shape or list of dimensions
    :param dim: dimension to rename
    :return: new shape/list with renamed dimension
    """
    return replace_dim(inp, anonymize_dim(dim), unanonymize_dim(dim))


def replace_dim(inp: typing.Union[DIM_LIST, mtf.Shape, mtf.Tensor],
                dim: typing.Union[mtf.Dimension, DIM_LIST],
                replaced: mtf.Dimension
                ) -> typing.Union[mtf.Shape, DIM_LIST, mtf.Tensor]:
    """
    Replace a dimension in a shape
    :param inp: shape or list of dimensions
    :param dim: dimension with the same name to replace it with
    :param replaced: dimension that will be replaced
    :return: new shape/list with changed dimension
    """
    shape = inp
    if isinstance(shape, mtf.Tensor):
        shape = shape.shape
    if isinstance(shape, mtf.Shape):
        shape = shape.dims
    if not check_for_dim(shape, replaced):
        return shape
    if not isinstance(dim, list):
        dim = [dim]
    out = []
    for cdim in shape:
        out.extend(dim if dim_name(replaced) == cdim.name else [cdim])
    if isinstance(inp, list):
        return out
    if isinstance(inp, mtf.Shape):
        return mtf.Shape(out)
    return reshape(inp, out)


def weighted_add(left: mtf.Tensor, right: mtf.Tensor, alpha: mtf.Tensor) -> mtf.Tensor:
    return add(multiply(left, alpha), multiply(right, add(1, negative(alpha))))


def utils_slice(tensor: mtf.Tensor, start: int, end: int, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    """
    Slice across a given (potentially non-anonymous) dimension in mtf.Tensor. This first anonymizes the dimension to
    allow slicing in the first place, next it slices across the dimension and only then it replicates it on all devices
    again.
    Non-Anonymous shapes are not necessary, as the anonymization can skip itself if it isn't necessary.
    :param tensor: mtf.Tensor to slice
    :param start: start of slice
    :param end: end of slice
    :param dim: dimension or name to slice in
    :return: slice of tensor
    """
    dim = dim_name(dim)
    if not start and get_dim(tensor, dim).size == end:
        return tensor
    return unanonymize(mtf_slice(anonymize(tensor, dim), start, end - start, anonymize_dim(dim)), dim)


def feature_dims_used(params: ModelParameter, shape: typing.Union[SHAPE, mtf.Tensor, mtf.Variable],
                      dims: OPT_DIMS = None) -> bool:
    if isinstance(shape, (mtf.Tensor, mtf.Variable)):
        shape = shape.shape
    if dims is None:
        dims = params.feature_dims + [anonymize_dim(dim) for dim in params.feature_dims]
        return bool(sum(f in dims_from_shape(shape) for f in dims) // 2)
    return all(f in dims_from_shape(shape) for f in dims)


def dims_from_shape(shape: ALL_SHAPES) -> DIM_LIST:
    if isinstance(shape, (mtf.Tensor, mtf.Variable)):
        shape = shape.shape
    if isinstance(shape, mtf.Shape):
        shape = shape.dims
    return shape


def shape_size(shape: ALL_SHAPES):
    return np.prod([d.size for d in dims_from_shape(shape)])


def get_intermediate(args: BlockArgs):
    if 'group' not in args:
        return args.params.intermediate
    return [args.params.head_dim,
            anonymize_dim(args.params.key_dim, args.params.key_dim.size * args.params.group_linear_factor)]


def linear_shapes(args: BlockArgs) -> LINEAR_SHAPES:
    features = get_intermediate(args) + args.params.feature_dims
    if 'group' in args and args.params.intermediate[-1] in args.tensor.shape:
        features.remove(args.params.key_dim)
        features.extend(args.params.intermediate)
    features = mtf.Shape(deduplicate(features))
    old = shape_crossection(args.tensor.shape, features)
    new = features - (old - ([args.params.head_dim] if 'group' in args and args.params.head_dim in old else []))
    return LINEAR_SHAPES(old.dims, new.dims)


def shape_crossection(*shapes: ALL_SHAPES):
    shapes = [dims_from_shape(s) for s in shapes]
    out = [dim for dim in shape_addition(*shapes) if all(dim in shape for shape in shapes)]
    return mtf.Shape(out)


def shape_addition(*shapes: ALL_SHAPES):
    dims = []
    for s in shapes:
        dims.extend(dims_from_shape(s))
    return mtf.Shape(deduplicate(dims))


def missing_dims(self: ALL_SHAPES, other: ALL_SHAPES):
    return (mtf.Shape(dims_from_shape(other)) - mtf.Shape(dims_from_shape(self))).dims


def compare_range(params: ModelParameter, dim0: mtf.Dimension, dim1: mtf.Dimension, comparison: typing.Callable):
    with tf1.variable_scope(f"compare{dim0.name}_{dim1.name}"):
        return cast(comparison(mtf_range(params.mesh, dim0, tf.int32),
                               mtf_range(params.mesh, dim1, tf.int32)),
                    params.variable_dtype.activation_dtype)


def get_attention_dim(args: BlockArgs) -> ATTENTION_DIM:
    attention_dims = (args.tensor.shape - args.params.feature_dims - args.params.intermediate)[1:]
    idx = args.params.attention_idx % len(attention_dims)
    dim = attention_dims[idx]
    return ATTENTION_DIM(idx, dim)


def is_masked(args: BlockArgs):
    return get_attention_dim(args).index in args.params.masked_attention_dimensions


def get_fan_in(params: ModelParameter, shape: ALL_SHAPES) -> DIM_LIST:
    shape = dims_from_shape(shape)
    features_used = feature_dims_used(params, shape)
    if features_used and shape.index(params.key_dim) == len(shape):
        return shape[:-2]
    if features_used:
        return shape[:2]
    return shape[:1]


# The majority of this Function was copied from:
# 'https://github.com/tensorflow/mesh/blob/8931eb9025f833b09d8425404ebd5801acbb0cac/mesh_tensorflow/ops.py#L5956-L6104'
# Copyright 2021 The Mesh TensorFlow Authors.
class WhileLoopWithControlDependencies(mtf.Operation):
    """While loop, like tf.while_loop."""

    def __init__(self, cond_fn: typing.Callable, body_fn: typing.Callable, inputs: typing.List[mtf.Tensor],
                 control_dependencies: typing.List[mtf.Operation] = None, tf_kwargs: typing.Dict = None,
                 has_accumulators: bool = False, name: str = "custom_while_loop"):
        """Create a WhileLoopOperation.
        A few differences from tf.while_loop:
        - gradients are not yet supported
        - inputs must be a list of tensors, as opposed to an arbitrary nested
          structure.  cond_fn and body_fn take an argument list
        - we support optional "accumulators" which are additional outputs
          returned by body_fn.  These are summed across all iterations and
          retured as additional outputs of the while-loop.  To use accumulators,
          the has_accumulators argument must be True.  For better performance,
          we delay allreduce on the accumulators until after the loop, so that it
          only needs to happen once.  This is useful, for example, if the
          accumulators are summing gradients for many mini-batches.
        Args:
          cond_fn: a function from n mtf Tensors to mtf Scalar
          body_fn: a function from n mtf Tensors to sequence of mtf Tensors
          inputs: list of n mtf Tensors
          control_dependencies: a list of mtf Operations
          tf_kwargs: a dictionary of arguments for tf.while_loop
          has_accumulators: a boolean
          name: a string
        Returns:
          a WhileLoopOperation
        """

        super(WhileLoopWithControlDependencies, self).__init__(inputs, mesh=inputs[0].mesh, name=name)
        self._cond_fn = cond_fn
        self._body_fn = body_fn
        self._tf_kwargs = tf_kwargs or {}
        self._control_dependencies = control_dependencies
        self._has_accumulators = has_accumulators
        assert not self._tf_kwargs.get("back_prop", False)

        # remove self from the graph's operations
        self.graph.operations.pop()
        before = len(self.graph.operations)

        def make_placeholders(name):
            return [mtf.Tensor(self, t.shape, t.dtype, name="%s:%d" % (name, i)) for i, t in enumerate(inputs)]

        self._cond_inputs = make_placeholders("cond_input")
        self._cond_output = self._cond_fn(*self._cond_inputs)
        self._cond_ops = self.graph.operations[before:]
        del self.graph.operations[before:]
        self._body_inputs = make_placeholders("body_input")
        _body_outputs = self._body_fn(*self._body_inputs)

        if type(_body_outputs) is dict:
            self._body_outputs = _body_outputs['outputs']
            self._control_dependencies = _body_outputs['control_dependencies']
        else:
            self._body_outputs = _body_outputs

        if len(self._body_outputs) < len(inputs):
            raise ValueError("body_fn produces fewer outputs than inputs")
        if len(self._body_outputs) > len(inputs) and not self._has_accumulators:
            raise ValueError("body_fn produces more outputs than inputs")
        for (i, (inp, body_out)) in enumerate(zip(inputs, self._body_outputs[:len(inputs)])):
            if inp.shape != body_out.shape:
                raise ValueError("shape mismatch i=%d inp=%s body_out=%s" % (i, inp, body_out))

        # Pull new variables outside the loop.
        added_ops = self.graph.operations[before:]
        del self.graph.operations[before:]
        self._body_ops = []
        for op in added_ops:
            if isinstance(op, mtf.Variable):
                self.graph.operations.append(op)
            else:
                self._body_ops.append(op)

        # re-add self to graph's operations
        self.graph.operations.append(self)
        self._outputs = [mtf.Tensor(self, t.shape, t.dtype, name="output:%d" % i)
                         for i, t in enumerate(self._body_outputs)]

        # Rerun to take the new output into account.
        self._splittable_dims, self._unsplittable_dims = (self._initialize_all_dimensions_as_splittable())

    def lower(self, lowering):
        mesh_impl = lowering.mesh_impl(self)

        def tf_cond_fn(*tf_inputs):
            for tf_inp, mtf_inp in zip(tf_inputs[:len(self._cond_inputs)], self._cond_inputs):
                lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
            for op in self._cond_ops:
                with tf.name_scope(op.name):
                    op.lower(lowering)
            lowered_output = lowering.tensors[self._cond_output]
            ret = lowered_output.to_laid_out_tensor().tensor_list[0]
            return ret

        # This array keeps track of which lowered body-outputs have type
        # LazyAllreduceSum.  We treat these specially  - instead of
        # immediately converting to LaidOutTensor (executing the allreduce)
        # we sum across iterations first, then allreduce at the end.
        # When one of the body outputs is a LazyAllreduceSum, we put the
        #  LazyAllreduceSum object into this array for future reference.
        is_lazyallreducesum = [None] * len(self._outputs)

        def tf_body_fn(*tf_inputs):
            """Body function for tf.while_loop.
            Args:
              *tf_inputs: a list of tf.Tensor
            Returns:
              a list of tf.Tensor
            """
            for tf_inp, mtf_inp in zip(tf_inputs[:len(self._inputs)], self._body_inputs):
                lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
            for op in self._body_ops:
                with tf.name_scope(op.name):
                    op.lower(lowering)

            if self._control_dependencies is not None:
                lower_control_dependencies = [lowering.lowered_operation(op) for op in self._control_dependencies]

            ret = []
            for i, mtf_out in enumerate(self._body_outputs):
                lowered_out = lowering.tensors[mtf_out]
                if isinstance(lowered_out, mtf.LazyAllreduceSum):
                    is_lazyallreducesum[i] = lowered_out
                    lowered_out = lowered_out.laid_out_input.tensor_list
                else:
                    lowered_out = lowered_out.to_laid_out_tensor().tensor_list

                if self._control_dependencies is not None:
                    with tf.control_dependencies(lower_control_dependencies):
                        lowered_out = [tf.identity(low_out) for low_out in lowered_out]

                ret.append(lowered_out)

            # accumulators
            if self._has_accumulators:
                for i in range(len(self._inputs), len(self._outputs)):
                    ret[i] = [x + y for x, y in zip(ret[i], tf_inputs[i])]

            return ret

        lowered_inputs = []

        for t in self.inputs:
            lowered_inputs.append(lowering.tensors[t].to_laid_out_tensor().tensor_list)

        # accumulators get initial value 0
        for t in self._body_outputs[len(self.inputs):]:
            def slice_fn():
                return tf.zeros(mesh_impl.slice_shape(t.shape), dtype=t.dtype)

            lowered_inputs.append(mesh_impl.slicewise(slice_fn).tensor_list)

        tf_outs = tf.while_loop(tf_cond_fn,
                                tf_body_fn,
                                lowered_inputs,
                                back_prop=False,
                                **self._tf_kwargs)
        for i, (tf_out, mtf_out) in enumerate(zip(tf_outs, self._outputs)):
            out = mesh_impl.LaidOutTensor(tf_out)
            lazy = is_lazyallreducesum[i]
            if lazy:
                out = mtf.LazyAllreduceSum(
                    mesh_impl, out, lazy.mesh_axes, lazy.add_counter_fn)
            lowering.set_tensor_lowering(mtf_out, out)
