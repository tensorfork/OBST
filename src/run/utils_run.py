import mesh_tensorflow as mtf
import tensorflow as tf

from tensorflow.python.ops import summary_ops_v2 as summary, variables
from tensorflow.python.tpu import tpu

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
        gs = tf.cast(local_global_step, tf.int64)
        with tf.control_dependencies([summary.scalar(key, local_value[key], step=gs) for key in local_value.keys()]):
            return tf.identity(local_tf_loss)

    # Cast the global step to tf.int32, since
    # outside_compilation does not support tf.int64.
    return tpu.outside_compilation(_host_loss_summary, tf_loss, value, tf.cast(global_step, tf.int32))


def add_histogram(tf_loss, value, global_step):
    """Add all summaries."""

    def _host_loss_summary(local_tf_loss, local_value, local_global_step):
        """Add summary.scalar in host side."""
        gs = tf.cast(local_global_step, tf.int64)
        with tf.control_dependencies([summary.histogram(key, local_value[key], step=gs) for key in local_value.keys()]):
            return tf.identity(local_tf_loss)

    # Cast the global step to tf.int32, since
    # outside_compilation does not support tf.int64.
    return tpu.outside_compilation(_host_loss_summary, tf_loss, value, tf.cast(global_step, tf.int32))


def _import_tensor(params, tensor, shape, name):
    return mtf.import_laid_out_tensor(params.mesh, params.mesh_impl.LaidOutTensor([tensor]), shape, name)