import typing

import mesh_tensorflow as mtf
import numpy as np
import pytest
import tensorflow as tf

from src.dataclass import BlockArgs, ModelParameter
from src.model import basic

tf1 = tf.compat.v1

tf1.disable_v2_behavior()


class BaseTest:
    def __init__(self,
                 *args,
                 mesh_shape: typing.Union[None, list, str] = None,
                 layout_rules: typing.Union[None, list, str] = None,
                 devices: typing.Union[None, typing.List[str]] = None,
                 **kwargs):
        self.mesh_shape = [] if mesh_shape is None else mesh_shape
        self.layout_rules = [] if layout_rules is None else layout_rules
        self.devices = ["cpu:0"] if devices is None else devices

        self.session_config = tf1.ConfigProto()
        self.session_config.allow_soft_placement = True

    def _close_session(self):
        default_session = tf1.get_default_session()
        if default_session is not None:
            default_session.close()

    def build(self, graph: mtf.Graph, mesh: mtf.Mesh,
              *args, **kwargs) -> typing.Tuple[typing.List[mtf.Tensor], typing.Any]:
        pass

    def run(self, sess: tf1.Session, outputs: typing.List[tf.Tensor], args: typing.Any) -> None:
        pass

    def __call__(self, *args, **kwargs) -> None:
        self._close_session()

        with tf.Graph().as_default() as tf_graph, tf1.Session(config=self.session_config, graph=tf_graph) as sess:
            graph = mtf.Graph()
            mesh = mtf.Mesh(graph, "MESH")

            outputs, args = self.build(graph, mesh, *args, **kwargs)

            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(self.mesh_shape, self.layout_rules, self.devices)
            lowering = mtf.Lowering(graph, {mesh: mesh_impl})

            outputs = [lowering.export_to_tf_tensor(output) for output in outputs]

            sess.run(tf1.global_variables_initializer())
            sess.run(lowering.copy_masters_to_slices())

            self.run(sess, outputs, args)


class OperationTest(BaseTest):
    def __init__(self, calculation_dtype="bfloat16", storage_dtype="bfloat16", slice_dtype="float32",
                 n_embd_per_head=16, n_head=1, batch_size=1, n_ctx=1, *args, **kwargs):
        super(OperationTest, self).__init__(*args, **kwargs)
        params = {'calculation_dtype': calculation_dtype,
                  "slice_dtype": slice_dtype,
                  "storage_dtype": storage_dtype,
                  "scale_by_depth": False,
                  "n_embd_per_head": n_embd_per_head,
                  "n_head": n_head,
                  "train_batch_size": batch_size,
                  "n_ctx": n_ctx,
                  "n_blocks": 1}
        self.args = BlockArgs(ModelParameter(params), None, [''])
        self.args.params.layout = self.layout_rules
        self.args.params.mesh_shape = self.mesh_shape

    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        pass

    def _run(self, out: np.array) -> None:
        pass

    def build(self, graph: mtf.Graph, mesh: mtf.Mesh,
              *args, **kwargs) -> typing.Tuple[typing.List[mtf.Tensor], typing.Any]:
        params = self.args.params
        params.mesh = mesh
        params.graph = graph
        inp = mtf.random_normal(mesh, [params.batch_dim, params.sequence_dim] + params.feature_dims,
                                dtype=params.variable_dtype.activation_dtype)

        return [self._build(inp)], None

    def run(self, sess: tf1.Session, outputs: typing.List[tf.Tensor], args: typing.Any) -> None:
        self._run(sess.run(outputs)[0])


class ReZero(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.rezero(self.args(inp))

    @staticmethod
    def _run(out: np.array) -> None:
        assert np.all(out == 0)


class AllSumFeedForwardIn(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        params = self.args.params
        return basic.orthogonal_var(self.args, params.feature_dims + params.intermediate)

    def _run(self, out: np.array) -> None:
        params = self.args.params
        mean = np.mean(out)
        size = np.prod([d.size for d in (params.feature_dims + params.intermediate)])
        intermediate = np.prod([d.size for d in params.intermediate])
        min_fan = min(size // intermediate, intermediate)
        analytical_var = (min_fan * (1 - min_fan / size) ** 2 + (size - min_fan) * (min_fan / size) ** 2) / size
        assert np.isclose(np.std(out), analytical_var ** 0.5, 0.03) and np.abs(mean) < 0.05


@pytest.mark.parametrize("test", [ReZero])
@pytest.mark.parametrize("calculation_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("storage_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("slice_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("embd_per_head", [1, 8, 64])
@pytest.mark.parametrize("n_head", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("n_ctx", [1, 8, 64])
def op_test(test: typing.Type, calculation_dtype: str, storage_dtype: str, slice_dtype: str, embd_per_head: int,
            n_head: int, batch_size: int, n_ctx: int):
    test(calculation_dtype=calculation_dtype, storage_dtype=storage_dtype, slice_dtype=slice_dtype,
         n_embd_per_head=embd_per_head, n_head=n_head, batch_size=batch_size, n_ctx=n_ctx)()
