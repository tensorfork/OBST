import typing

import mesh_tensorflow as mtf
import numpy as np
import pytest
import tensorflow as tf

from backend import OperationTest, RELU_STD
from src.model import basic

tf1 = tf.compat.v1


class ReZero(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.rezero(self.args(inp))

    @staticmethod
    def _run(out: np.array) -> None:
        assert np.all(out == 0)


class Dropout(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.dropout(self.args(inp)([f'dropout_rate{self.args.params.input_dropout}']))

    def _run(self, out: np.array) -> None:
        self._is_close(np.sum(out == 0) / out.size, self.args.params.input_dropout, 0.2)


class Activate(OperationTest):
    @staticmethod
    def _activation() -> str:
        return ''

    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.activate(self.args(inp)([self._activation()]))


class Identity(Activate):
    def _run(self, out: np.array) -> None:
        self._is_close(np.std(out), 1)


class ReLU(Activate):
    @staticmethod
    def _activation() -> str:
        return 'relu'

    def _run(self, out: np.array) -> None:
        self._is_close(np.std(out), RELU_STD, 0.2)


@pytest.mark.parametrize("test", [ReZero, Dropout, Identity, ReLU])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [64, 256, 1024])
@pytest.mark.parametrize("batch_size", [16, 256])
def pointwise_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str, embd_per_head: int,
                   batch_size: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         features_per_head=embd_per_head, heads=1, batch_size=batch_size, sequence_length=1)()
