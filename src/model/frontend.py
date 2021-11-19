import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .basic import dropout, feed_forward, rezero, group_linear, product_key_memory
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

    for idx, layer in enumerate(block_part_config.layer, 1):
        name, *extras = layer.split('-')
        args = BlockArgs(params, out, extras, idx == len(block_part_config.layer))
        out = scoped(name + '_', LAYER_FUNCTIONS[name], args)

    if block_part_config.skip and block_part_config.memory_reduction_strategy in ("none", "checkpoint"):
        out = add(out, block_input)
    return out


def block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor,
                  name_prefix: str = 'block') -> mtf.Tensor:
    return scoped(f"{name_prefix}_", _get_block_part, block_part_config, params, block_input)


def split_path(args: BlockArgs) -> mtf.Tensor:
    base, *name_extras = [path for path in '-'.join(args.name_extras).split(';')]
    base = base.split('-')
    if 'add' in base:
        out = 0
        fn = add
    elif 'multiply' in base:
        out = 1
        fn = multiply
    else:
        raise ValueError

    for idx, conf in enumerate(name_extras):
        out = fn(out, _get_block_part(BlockConfig({'skip': False, 'layer': conf.split(',')}, ''),
                                      args.params, args.tensor))

    return out


LAYER_FUNCTIONS = {'feed_forward': feed_forward,
                   'attention': attention,
                   'norm': norm,
                   'rezero': rezero,
                   'activation': activate,
                   'convolution': convolution,
                   'dropout': dropout,
                   'group_linear': group_linear,
                   'split_path': split_path,
                   'product_key_memory': product_key_memory
                   }
