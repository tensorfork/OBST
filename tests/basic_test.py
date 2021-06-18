import typing

import mesh_tensorflow as mtf
import numpy as np
import pytest
import tensorflow as tf

from backend import OperationTest, curry_class
from src.model import basic

tf1 = tf.compat.v1

tf1.disable_v2_behavior()


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
        params = self.args.params
        self._is_close(np.sum(out == 0) / out.size, params.input_dropout)


@pytest.mark.parametrize("test",
                         [ReZero,
                          curry_class(Dropout, input_dropout=0), curry_class(Dropout, input_dropout=0.1),
                          curry_class(Dropout, input_dropout=0.25), curry_class(Dropout, input_dropout=0.5)])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [1, 16, 256])
@pytest.mark.parametrize("batch_size", [1, 8, 64, 256])
def op_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str, embd_per_head: int,
            batch_size: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         n_embd_per_head=embd_per_head, n_head=1, batch_size=batch_size, n_ctx=1)()
