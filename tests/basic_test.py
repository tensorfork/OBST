import typing

import jsonpickle
import mesh_tensorflow as mtf
import numpy as np
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
        with open("tests/test_config.json", 'r') as f:
            config = f.read()
        config = jsonpickle.loads(config)
        params = ModelParameter(config)
        self.args = BlockArgs(params, None, [''])
        params.mesh_shape = self.mesh_shape = [] if mesh_shape is None else mesh_shape
        params.layout = self.layout_rules = [] if layout_rules is None else layout_rules
        self.devices = ["cpu:0"] if devices is None else devices

        self.session_config = tf1.ConfigProto()
        self.session_config.allow_soft_placement = True

    def _close_session(self):
        default_session = tf1.get_default_session()
        if default_session is not None:
            default_session.close()

    def build(self, inp: mtf.Tensor, *args, **kwargs) -> typing.Tuple[typing.List[mtf.Tensor], typing.Any]:
        pass

    def run(self, sess: tf1.Session, outputs: typing.List[tf.Tensor], args: typing.Any) -> None:
        pass

    def __call__(self,
                 dim_size: int,
                 dim_count: int,
                 *args,
                 **kwargs) -> None:
        self._close_session()

        with tf.Graph().as_default() as tf_graph, tf1.Session(config=self.session_config, graph=tf_graph) as sess:
            self.args.params.graph = graph = mtf.Graph()
            self.args.params.mesh = mesh = mtf.Mesh(graph, "MESH")

            inp = tf1.random.normal(shape=[dim_size] * dim_count,
                                    dtype=self.args.params.variable_dtype.activation_dtype)
            mtf_shape = [mtf.Dimension(str(i), dim_size) for i in range(dim_count)]
            outputs, args = self.build(mtf.import_tf_tensor(mesh, inp, shape=mtf_shape), *args, **kwargs)

            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(self.mesh_shape, self.layout_rules, self.devices)
            lowering = mtf.Lowering(graph, {mesh: mesh_impl})

            outputs = [lowering.export_to_tf_tensor(output) for output in outputs]

            sess.run(tf1.global_variables_initializer())
            sess.run(lowering.copy_masters_to_slices())

            self.run(sess, outputs, args)


class OperationTest(BaseTest):
    def _build(self, inp: mtf.Tensor, *args, **kwargs) -> mtf.Tensor:
        pass

    def _run(self, out: np.array) -> None:
        pass

    def build(self, inp: mtf.Tensor, *args, **kwargs) -> typing.Tuple[typing.List[mtf.Tensor], typing.Any]:
        return [self._build(inp, *args, **kwargs)], None

    def run(self, sess: tf1.Session, outputs: typing.List[tf.Tensor], args: typing.Any) -> None:
        self._run(sess.run(outputs)[0])


class ReZero(OperationTest):
    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        return basic.rezero(self.args(inp))

    @staticmethod
    def _run(out: np.array) -> None:
        assert np.all(out == 0)


def rezero_test():
    ReZero()(1, 1)
