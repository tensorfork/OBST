import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer

from .dataclass import BlockArgs, ModelParameter
from .mtf_wrapper import cast, mtf_range, random_name
from .utils_core import default

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


def unanonymize(inp: mtf.Tensor, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
    """
    Inverse of anonymize. Un-replicates tensor across axis by removing the underscore from the name of a dimension of
    the tensor. This allows mtf to split the tensor across a given dimension again.
    :param inp: tensor to replicate
    :param dim: dimension of tensor
    :return: un-replicated tensor
    """
    dim = anonymize_dim(dim)
    if not check_for_dim(inp, dim):
        return inp
    return mtf.rename_dimension(inp, dim, dim_name(unanonymize_dim(dim)))


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


def get_dim(shape: typing.Union[mtf.Tensor, mtf.Shape, typing.List[mtf.Dimension]],
            dim: typing.Union[mtf.Dimension, str],
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
    return unanonymize(mtf.concat([anonymize(t, dim) for t in tensor_list], anonymize_dim(dim)), dim)


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
    return mtf.pad(anonymize(tensor, dim), padding, anonymize_dim(dim))


def to_fp32(tensor: mtf.Tensor) -> mtf.Tensor:
    """
    Cast a tensor to float
    :param tensor: tensor to be casted
    :return: casted tensor
    """
    return mtf.cast(tensor, tf.float32)


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
            return mtf.reshape(inp, shape)
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

mtf.reduce_sum()
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


def get_variable(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer, trainable: bool):
    full_name = f'{tf1.get_variable_scope().name}/{name}'
    if full_name in params.mesh.graph.name_to_variable:
        return params.mesh.graph.name_to_variable[full_name].outputs[0]
    shape = deduplicate(mtf.Shape(shape))
    var = mtf.Variable(params.mesh, name, shape, params.variable_dtype, initializer, trainable)
    params.mesh.graph.name_to_variable[full_name] = var
    return var.outputs[0]


def non_replicated_variable(params: ModelParameter, name: str, shape: SHAPE, initializer: Initializer, trainable: bool):
    var = get_variable(params, name, shape, initializer, trainable)
    if not params.grad_accumulation or params.batch_splits == 1:
        return var
    return non_replicated_broadcast(var, [params.batch_dim] + dims_from_shape(shape))


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
    return mtf.reshape(inp, out)


def weighted_add(left: mtf.Tensor, right: mtf.Tensor, alpha: mtf.Tensor) -> mtf.Tensor:
    return left * alpha + right * (1 - alpha)


def slice(tensor: mtf.Tensor, start: int, end: int, dim: typing.Union[mtf.Dimension, str]) -> mtf.Tensor:
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
    return unanonymize(mtf.slice(anonymize(tensor, dim), start, end - start, anonymize_dim(dim)), dim)


def feature_dims_used(params: ModelParameter, shape: typing.Union[SHAPE, mtf.Tensor, mtf.Variable],
                      dims: OPT_DIMS = None) -> bool:
    if isinstance(shape, (mtf.Tensor, mtf.Variable)):
        shape = shape.shape
    if dims is None:
        dims = params.feature_dims
    return all(f in shape for f in dims)


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
    features = mtf.Shape(deduplicate(get_intermediate(args) + args.params.feature_dims))
    if 'group' in args:
        features -= [args.params.head_dim]
    old = shape_crossection(args.tensor.shape, features).dims
    return LINEAR_SHAPES(old, (features - old).dims)


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
