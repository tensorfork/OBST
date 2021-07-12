import typing

import jsonpickle
import mesh_tensorflow as mtf
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.tpu import tpu

from ..dataclass import ModelParameter
from ..mtf_wrapper import import_laid_out_tensor
from ..utils_core import color_print
from .. import tf_wrapper as tfw

tf1 = tf.compat.v1
Dataset = tf1.data.Dataset


class CheckpointLoaderHook(tf.estimator.SessionRunHook):
    """Load checkpoint right after the session started."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def after_create_session(self, session, coord):
        saver_collection = tf1.get_collection(tf1.GraphKeys.SAVERS)
        if saver_collection:
            check_point = tf.train.latest_checkpoint(self.checkpoint_dir)
            if check_point:
                saver_collection[0].restore(session, check_point)


def add_summary(tf_loss, value, global_step):
    """Add all summaries."""

    def _host_loss_summary(local_tf_loss, local_value, local_global_step):
        """Add summary.scalar in host side."""
        gs = tfw.cast(local_global_step, tf.int64)
        with tfw.control_dependencies([summary.scalar(key, local_value[key], step=gs) for key in local_value.keys()]):
            return tfw.identity(local_tf_loss)

    # Cast the global step to tf.int32, since
    # outside_compilation does not support tf.int64.
    return tpu.outside_compilation(_host_loss_summary, tf_loss, value, tfw.cast(global_step, tf.int32))


def add_histogram(tf_loss, value, global_step):
    """Add all summaries."""

    def _host_loss_summary(local_tf_loss, local_value, local_global_step):
        """Add summary.scalar in host side."""
        gs = tfw.cast(local_global_step, tf.int64)
        with tfw.control_dependencies([summary.histogram(key, local_value[key], step=gs)
                                       for key in local_value.keys()]):
            return tfw.identity(local_tf_loss)

    # Cast the global step to tf.int32, since
    # outside_compilation does not support tf.int64.
    return tpu.outside_compilation(_host_loss_summary, tf_loss, value, tfw.cast(global_step, tf.int32))


def _import_tensor(params: ModelParameter, tensor, shape, name):
    return import_laid_out_tensor(params, params.mesh_impl.LaidOutTensor([tensor]), shape, name)


def analyze_model(params: ModelParameter, time_to_build: float, graph: mtf.Graph):
    color_print(params, f"Built in {time_to_build:.1f}s")
    param_count = int(sum([variable.size for variable in graph.trainable_variables]))
    var_count = int(sum([variable.size for variable in graph.all_variables]))
    embed_param_count = int(sum([variable.size for variable in
                                 graph.trainable_variables if 'embed' in variable.name]))
    body_param_count = int(sum([variable.size for variable in
                                graph.trainable_variables if 'body' in variable.name]))

    print('')

    constant = '  variables: '
    variable_mapping = [('Model', param_count - embed_param_count),
                        ('Embedding', embed_param_count),
                        ('Body with Embed', body_param_count),
                        ('Untrainable', var_count - param_count),
                        ('', 0),
                        ('Total trainable', param_count),
                        ('Total', var_count)]
    variable_mapping = [(name, f'{int(count):,}') for name, count in variable_mapping]
    max_str = max(len(name) for name, _ in variable_mapping)
    max_int = max(len(count) for _, count in variable_mapping)
    for name, count in variable_mapping:
        if not name:
            color_print(params, '-' * (max_str + max_int + len(constant)))
            continue
        color_print(params, f'{name:<{max_str}s}{constant}{count:>{max_int}s}')

    color_print(params, "\nDimensions:")
    for dim_name in sorted(list(set([item for variable in graph.all_variables
                                     for item in variable.shape.dimension_names]))):
        color_print(params, dim_name)
    print('')

    model_size = {'model_variables': int(param_count - embed_param_count),
                  'embedding_variables': int(embed_param_count),
                  'body_variables': int(body_param_count),
                  'untrainable_variables': int(var_count - param_count),
                  'total_trainable_variables': int(param_count),
                  'total_variables': int(var_count)
                  }

    if params.train:
        size_dump = jsonpickle.dumps(model_size, indent=4)
        with tf.io.gfile.GFile(f"{params.model_path}/model_size.info", 'w') as f:
            f.write(size_dump)


def rep_batch(params: ModelParameter, shape: [mtf.Shape, typing.List[mtf.Dimension]]):

    if params.macro_batching > 1 and params.train:
        return mtf.replace_dimensions(shape, params.batch_dim, params.macro_batch_dim)
    return shape