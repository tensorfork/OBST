import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .basic import dropout, feed_forward, rezero, group_linear
from .convolution import convolution
from .normalization import norm
from .spatial import attention
from ..dataclass import BlockArgs, BlockConfig, ModelParameter
from ..mtf_wrapper import add, multiply
from ..utils_core import scoped

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def _get_block_part(block_part_config: BlockConfig, params: ModelParameter, block_input: mtf.Tensor) -> mtf.Tensor:
    out = block_input
    for layer in block_part_config.layer:
        name, *extras = layer.split('-')
        out = scoped(name + '_', LAYER_FUNCTIONS[name], BlockArgs(params, out, extras))

    if block_part_config.skip and block_part_config.memory_reduction_strategy in ("none", "checkpoint"):
        out = add(out, block_input)
    return out


def block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor,
                  name_prefix: str = 'block') -> mtf.Tensor:
    return scoped(f"{name_prefix}_", _get_block_part, block_part_config, params, block_input)


def split_path(args: BlockArgs) -> mtf.Tensor:
    base, *name_extras = [[block.split('-') for block in path.split(',')]
                          for path in '-'.join(args.name_extras).split(';')]
    if 'add' in base:
        out = 0
        fn = add
    elif 'multiply' in base:
        out = 1
        fn = multiply
    else:
        raise ValueError

    for n in name_extras:
        name, *extras = n
        out = fn(scoped(name, LAYER_FUNCTIONS[name], args(extras)), out)

    return out


LAYER_FUNCTIONS = {'feed_forward': feed_forward,
                   'attention': attention,
                   'norm': norm,
                   'rezero': rezero,
                   'activation': activate,
                   'convolution': convolution,
                   'dropout': dropout,
                   'group_linear': group_linear,
                   'split_path': split_path
                   }
