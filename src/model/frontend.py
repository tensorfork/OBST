import typing

import mesh_tensorflow as mtf
import tensorflow as tf

from .activation import activate
from .attention import attention
from .basic import dropout, embed, feed_forward, norm, rezero
from .convolution import convolution
from ..dataclass import BlockConfig, ModelParameter
from ..mtf_wrapper import scoped
from ..utils_core import random_name

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1

LAYER_FUNCTIONS = {'feed_forward': feed_forward,
                   'attention':    attention,
                   'norm':         norm,
                   'rezero':       rezero,
                   'activation':   activate,
                   'convolution':  convolution,
                   'dropout':      dropout
                   }


def block_part_fn(params: ModelParameter, block_part_config: BlockConfig, block_input: mtf.Tensor,
                  name_prefix: str = 'block') -> mtf.Tensor:
    out = block_input
    with tf1.variable_scope(random_name(f"{name_prefix}_")):
        for layer in block_part_config.layer:
            name, *extras = layer.split('-')
            out = scoped(name, LAYER_FUNCTIONS[name], params, out, extras)

        if not block_part_config.use_revnet and block_part_config.skip:
            out += block_input

    return out
