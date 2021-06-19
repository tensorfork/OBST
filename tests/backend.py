import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

from src.dataclass import BlockArgs, ModelParameter

tf1 = tf.compat.v1

tf1.disable_v2_behavior()

RELU_STD = 1 / 1.42


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
    def __init__(self, **kwargs):
        super(OperationTest, self).__init__(**kwargs)
        params = ModelParameter(kwargs)
        self.fp16 = "16" in (kwargs['calculation_dtype'] + kwargs['slice_dtype'] + kwargs['storage_dtype'])
        self.args = BlockArgs(params, None, [''])
        self.args.params.layout = self.layout_rules
        self.args.params.mesh_shape = self.mesh_shape
        self.tolerance = 1 / (params.train_batch_size * params.n_ctx * params.n_embd) ** (0.05 if self.fp16 else 1 / 3)

    def _build(self, inp: mtf.Tensor) -> mtf.Tensor:
        pass

    def _run(self, out: np.array) -> None:
        pass

    def _is_close(self, x: np.array, y: np.array, rtol: float = 1e-3):
        assert np.isclose(x, y, rtol, self.tolerance)

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


def curry_class(base: typing.Type, **kwargs) -> typing.Callable:
    def _fn(**kw):
        return base(**kw, **kwargs)

    _fn.__name__ = f'{base.__name__}({",".join(f"{k}={v}" for k, v in kwargs.items())})'
    return _fn
