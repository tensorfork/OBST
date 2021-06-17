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
                 n_embd_per_head=16, n_head=1, *args, **kwargs):
        super(OperationTest, self).__init__(*args, **kwargs)
        params = {'calculation_dtype': calculation_dtype,
                  "slice_dtype": slice_dtype,
                  "storage_dtype": storage_dtype,
                  "n_embd_per_head": n_embd_per_head,
                  "n_head": n_head}
        self.args = BlockArgs(ModelParameter(params), None, [''])
        self.args.params.layout = self.layout_rules
        self.args.params.mesh_shape = self.mesh_shape

    def _build(self, inp: mtf.Tensor, *args, **kwargs) -> mtf.Tensor:
        pass

    def _run(self, out: np.array) -> None:
        pass

    def build(self, graph: mtf.Graph, mesh: mtf.Mesh, dim_size: int, dim_count: int,
              *args, **kwargs) -> typing.Tuple[typing.List[mtf.Tensor], typing.Any]:
        self.args.params.mesh = mesh
        self.args.params.graph = graph
        print(np.prod([dim_size] * dim_count))
        inp = tf1.random.normal(shape=[dim_size] * dim_count,
                                dtype=self.args.params.variable_dtype.activation_dtype)
        mtf_shape = [mtf.Dimension(str(i), dim_size) for i in range(dim_count)]
        inp = mtf.import_tf_tensor(mesh, inp, shape=mtf_shape)

        return [self._build(inp, *args, **kwargs)], None

    def run(self, sess: tf1.Session, outputs: typing.List[tf.Tensor], args: typing.Any) -> None:
        self._run(sess.run(outputs)[0])


class ReZero(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.rezero(self.args(inp))

    @staticmethod
    def _run(out: np.array) -> None:
        assert np.all(out == 0)


@pytest.mark.parametrize("test", [ReZero])
@pytest.mark.parametrize("dim_size,dim_count",
                         [(1, 1),
                          (1, 16),
                          (16, 1),
                          (2, 8),
                          (8, 4),
                          (64, 1),
                          (64, 2),
                          (8192, 1)])
def op_test(test: typing.Type, dim_size: int, dim_count: int):
    test()(dim_size, dim_count)
