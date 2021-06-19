import typing

import mesh_tensorflow as mtf
import numpy as np
import pytest
import tensorflow as tf

from backend import OperationTest
from src.model import basic

tf1 = tf.compat.v1

tf1.disable_v2_behavior()

RELU_STD = 1 / 1.42


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
        self._is_close(np.std(out), RELU_STD)


class Linear(OperationTest):
    @staticmethod
    def _activation() -> str:
        return ''

    @staticmethod
    def _target_std() -> float:
        return 1

    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        for _ in range(self.args.params.train_steps):
            inp = basic.activate(self.args(basic.wrapped_linear(self.args(inp)))([self._activation()]))
        return inp

    def _run(self, out: np.array) -> None:
        self.tolerance *= self.args.params.train_steps ** 0.5
        target_std = self._target_std()
        if self.args.params.scale_by_depth:
            target_std /= self.args.params.n_blocks ** 0.5
        self._is_close(np.mean(np.std(out, -1)), target_std, 0.2)


class ReLULinear(Linear):
    @staticmethod
    def _activation() -> str:
        return 'relu'

    def _target_std(self) -> float:
        return RELU_STD ** self.args.params.train_steps


@pytest.mark.parametrize("test", [ReZero, Dropout, Identity, ReLU])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [1, 16, 256])
@pytest.mark.parametrize("batch_size", [1, 8, 64, 256])
def pointwise_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str, embd_per_head: int,
                   batch_size: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         n_embd_per_head=embd_per_head, n_head=1, batch_size=batch_size, n_ctx=1)()


@pytest.mark.parametrize("test", [Linear, ReLULinear])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [16, 256])
@pytest.mark.parametrize("heads", [1, 2, 8])
@pytest.mark.parametrize("scale_by_depth", [True, False])
@pytest.mark.parametrize("train_steps", [1, 2, 8])
def square_matmul_std_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str,
                           embd_per_head: int, heads: int, scale_by_depth: bool, train_steps: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         n_embd_per_head=embd_per_head, n_head=heads, batch_size=1, n_ctx=1, group_linear_factor=heads,
         scale_by_depth=scale_by_depth, train_steps=train_steps)()
