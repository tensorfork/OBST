import math
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from .activation import activate_util
from .backend import get_intermediate, get_variable, linear_from_features, linear_to_features, normal_var
from ..dataclass import ModelParameter
from ..mtf_wrapper import (add_n, dropout as utils_dropout, einsum, mtf_range, sigmoid)
from ..utils_core import random_name
from ..utils_mtf import DIM_LIST, SHAPE, shape_size

ATTENTION_DIM = typing.NamedTuple("AttentionDim", (('index', int), ('dim', mtf.Dimension)))

tf1 = tf.compat.v1


def rezero(params, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    return block_input * get_variable(params, [], tf.constant_initializer(0))




def dropout(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]):
    keep = 1
    for extra in name_extras:
        if extra.startswith('dropout_rate'):
            keep = 1 - float(extra[len('dropout_rate'):])
    return utils_dropout(block_input, keep)


def feed_forward(params: ModelParameter, block_input: mtf.Tensor, name_extras: typing.List[str]) -> mtf.Tensor:
    intermediate = get_intermediate(params, name_extras)

    def _from_feat():
        return linear_from_features(params, block_input, intermediate)

    mid = dropout(params, activate_util(name_extras, _from_feat()), name_extras)
    if 'glu' in name_extras or 'glu_add' in name_extras:
        mid *= sigmoid(_from_feat())
    if 'glu_add' in name_extras:
        mid += activate_util(name_extras, _from_feat())
    return linear_to_features(params, mid, intermediate)
