import typing

import mesh_tensorflow as mtf
import numpy as np
import pytest
import tensorflow as tf

from src.model import backend
from src.utils_mtf import get_intermediate, deduplicate
from backend import OperationTest

tf1 = tf.compat.v1

tf1.disable_v2_behavior()


class VariableCheck(OperationTest):
    def _in_dims(self) -> typing.List[mtf.Dimension]:
        return []

    def _out_dims(self) -> typing.List[mtf.Dimension]:
        return []

    def _shape(self) -> typing.List[mtf.Dimension]:
        return deduplicate(self._in_dims() + self._out_dims())

    @staticmethod
    def _target_std() -> float:
        return 0

    @staticmethod
    def _target_mean() -> float:
        return 0

    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return mtf.zeros(inp.mesh, self._shape())

    def _run(self, out: np.array) -> None:
        relative_tolerance = 1 / np.prod([d.size for d in self._shape()]) ** (0.05 if self.fp16 else 0.5)
        assert np.isclose(np.std(out), self._target_std(), 2 * relative_tolerance)
        assert np.isclose(np.mean(out), self._target_mean(), 1e-3, 1 * relative_tolerance)


class NormalCheck(VariableCheck):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return backend.normal_var(self.args, self._shape(), self._target_std(), self._target_mean())


class NormShiftCheck(VariableCheck):
    @staticmethod
    def _target_std() -> float:
        return 0.02

    @staticmethod
    def _target_mean() -> float:
        return 0


class NormScaleCheck(VariableCheck):
    @staticmethod
    def _target_std() -> float:
        return 0.02

    @staticmethod
    def _target_mean() -> float:
        return 1


class EmbeddingCheck(VariableCheck):
    def _target_std(self) -> float:
        return self.args.params.embedding_stddev

    @staticmethod
    def _target_mean() -> float:
        return 0


class OrthogonalCheck(VariableCheck):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return backend.orthogonal_var(self.args, self._shape())

    def _target_std(self) -> float:
        size = np.prod([d.size for d in self._shape()])
        feature_dims = self.args.params.feature_dims
        inp = feature_dims if self._in_dims() == feature_dims or self._out_dims() == feature_dims else self._in_dims()
        intermediate = np.prod([d.size for d in inp])
        min_fan = min(size // intermediate, intermediate)
        std = ((min_fan * (1 - min_fan / size) ** 2 + (size - min_fan) * (min_fan / size) ** 2) / size) ** 0.5
        if not self.args.params.scale_by_depth:
            return std
        return std / self.args.params.n_blocks ** 0.5


class AllSumFeedForwardIn(OrthogonalCheck):
    def _in_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.feature_dims

    def _out_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.intermediate


class AllSumFeedForwardOut(OrthogonalCheck):
    def _in_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.intermediate

    def _out_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.feature_dims


class GroupFeedForwardIn(OrthogonalCheck):
    def _in_dims(self) -> typing.List[mtf.Dimension]:
        return get_intermediate(self.args(['group']))

    def _out_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.feature_dims


class GroupFeedForwardOut(OrthogonalCheck):
    def _in_dims(self) -> typing.List[mtf.Dimension]:
        return self.args.params.feature_dims

    def _out_dims(self) -> typing.List[mtf.Dimension]:
        return get_intermediate(self.args(['group']))


class SharedOrthogonalVariable(GroupFeedForwardIn):
    def _get_shared_var(self, idx: int) -> mtf.Tensor:
        with tf1.variable_scope(f"gpt/body/{self.args.params.attention_idx}_0/feed_forward_{idx}/"):
            out = backend.orthogonal_var(self.args(['shared']), self._shape())
            self.args.params.attention_idx += idx == 0
            return out

    def _run(self, out: np.array) -> None:
        assert all(np.array_equal(out[0], out[i]) for i in range(1, out.shape[0]))


class SingleSharedVariable(SharedOrthogonalVariable):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return mtf.stack([self._get_shared_var(0) for _ in range(self.args.params.n_blocks)], "items")


class DoubleSharedVariable(SharedOrthogonalVariable):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return mtf.stack([mtf.stack([self._get_shared_var(0), self._get_shared_var(1)], "non_shared")
                          for _ in range(self.args.params.n_blocks)], "items")


def curry_class(base: typing.Type, **kwargs) -> typing.Callable:
    def _fn(**kw):
        return base(**kw, **kwargs)

    _fn.__name__ = f'{base.__name__}({",".join(f"{k}={v}" for k, v in kwargs.items())})'
    return _fn


@pytest.mark.parametrize("test",
                         [curry_class(AllSumFeedForwardIn, scale_by_depth=True),
                          curry_class(AllSumFeedForwardOut, scale_by_depth=True),
                          curry_class(GroupFeedForwardIn, scale_by_depth=True),
                          curry_class(GroupFeedForwardOut, scale_by_depth=True),
                          curry_class(AllSumFeedForwardIn, scale_by_depth=False),
                          curry_class(AllSumFeedForwardOut, scale_by_depth=False),
                          curry_class(GroupFeedForwardIn, scale_by_depth=False),
                          curry_class(GroupFeedForwardOut, scale_by_depth=False),
                          NormShiftCheck, NormScaleCheck, EmbeddingCheck, SingleSharedVariable, DoubleSharedVariable])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [1, 16, 256])
@pytest.mark.parametrize("n_head", [1, 4, 16])
def op_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str, embd_per_head: int,
            n_head: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         n_embd_per_head=embd_per_head, n_head=n_head, batch_size=1, n_ctx=1)()
