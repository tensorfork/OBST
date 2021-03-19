"""
Contains functions to create a training loop and log its outputs to tensorboard
"""

import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.ops import variables
from tensorflow.python.tpu import tpu_estimator

from .dataclass import ModelParameter
from .model import build
from .optimizers import get_optimizer


class CheckpointLoaderHook(tf.estimator.SessionRunHook):
    """Load checkpoint right after the session started."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def after_create_session(self, session, coord):
        saver_collection = tf.get_collection(tf.GraphKeys.SAVERS)
        if saver_collection:
            check_point = tf.train.latest_checkpoint(self.checkpoint_dir)
            if check_point:
                saver_collection[0].restore(session, check_point)


def create_host_call(model_dir: str) -> typing.Optional[typing.Tuple[typing.Callable, typing.List[tf.Tensor]]]:
    """Construct a host_call writing scalar summaries.
    Borrowed from t2t.
    Args:
        model_dir: String containing path to train
    Returns:
        (fn, args) Pair to be called by TPUEstimator as the host_call.
    """

    graph = tf.get_default_graph()
    # A list of (name, lowered tensor) tuples
    summaries = graph.get_collection(mtf.utils.SCALAR_SUMMARIES_COLLECTION_KEY)

    def maybe_cast(tensor: tf.Tensor) -> tf.Tensor:
        if tensor.shape.is_compatible_with([]):
            tensor = tf.reshape(tensor, [1])
        if tensor.dtype == tf.int64:
            return tf.cast(tensor, tf.int32)
        if tensor.dtype == tf.bfloat16 or tensor.dtype == tf.float16:
            return tf.cast(tensor, tf.float32)
        return tensor

    reshaped_tensors = [maybe_cast(t) for _, t in summaries]

    # When no supported summaries are found, don't create host_call. Otherwise,
    # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
    # it, eventually causing hang.
    if not reshaped_tensors:
        return None

    def host_call_fn(global_step, *args):
        """Training host call. Creates scalar summaries for training metrics."""
        # This function is executed on the CPU and should not directly reference
        # any Tensors in the rest of the `model_fn`. To pass Tensors from the
        # model to the `model_fn`, provide as part of the `host_call`.
        global_step = tf.cast(global_step[0], tf.int64)
        with tf2.summary.create_file_writer(model_dir).as_default():
            # We cannot directly use any tensor from summaries, because each
            # tensor here must be a concat of multiple tensors from all shards.
            # Therefore, we rely on the assumption that args wil have the same
            # length as summaries, and all tensors in args will have the same
            # order of self._tup_summaries.
            assert len(args) == len(summaries)
            for i, tensor in enumerate(args):
                name = summaries[i][0]
                tf2.summary.scalar(name, tf.reduce_mean(tensor), step=global_step)
        return tf.summary.all_v2_summary_ops()

    global_step_t = tf.reshape(tf.cast(tf.train.get_global_step(), tf.int32), [1])
    return host_call_fn, [global_step_t] + reshaped_tensors


def _import_tensor(params, tensor, shape, name):
    return mtf.import_fully_replicated(params.mesh, tensor, shape, name)
    # return mtf.import_laid_out_tensor(params.mesh, params.mesh_impl.LaidOutTensor([tensor]), shape, name)


def model_fn(features: typing.Dict[str, tf.Tensor], mode: str, params: dict):
    params = ModelParameter(params)
    manual_global_step = tf.get_variable("manual_global_step", [], tf.int64, initializer=tf.zeros_initializer(),
                                         trainable=False,
                                         aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
    # Construct mtf graph + mesh from params
    graph = mtf.Graph()

    # Build mtf mesh object
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(mesh_shape, mtf.convert_to_layout_rules(params.layout),
                                                [""] * mesh_shape.size, params.context.device_assignment)
    params.mesh = mtf.Mesh(graph, "mesh", mtf.utils.BalancedVariablePlacer(
            [params.context.tpu_host_placement_function(host_id=i) for i in range(params.context.num_hosts)]))
    params.mesh_impl = mesh_impl

    frame_input = None
    cat_mask_src = None
    cat_mask_tgt = None
    token_x_input = None
    token_y_input = None
    frame_mask_src = None
    frame_mask_tgt = None
    token_mask = None

    if params.use_video:
        frame_input = _import_tensor(params, features['frame'], params.frame_input_shape, "frame_input")
        cat_mask_src = _import_tensor(params, features['cat_mask_x'], params.frame_mask_shape, "cat_mask_x")
        cat_mask_tgt = _import_tensor(params, features['cat_mask_y'], params.frame_mask_shape, "cat_mask_y")
        frame_mask_src = _import_tensor(params, features['vid_msk_src'], params.frame_mask_shape, "vid_msk_src")
        frame_mask_tgt = _import_tensor(params, features['vid_msk_tgt'], params.frame_mask_shape, "vid_msk_tgt")

        if params.use_language:
            token_x_input = _import_tensor(params, features['token_x'], params.token_dim_shape, "tkn_src")
            token_y_input = _import_tensor(params, features['token_y'], params.token_dim_shape, "tkn_tgt")
            token_mask = _import_tensor(params, features['txt_msk'], params.token_dim_shape, "txt_msk")

    else:  # params.use_language
        token_x_input = _import_tensor(params, features['token_x'], params.token_dim_shape, "tkn_src")
        token_y_input = _import_tensor(params, features['token_y'], params.token_dim_shape, "tkn_tgt")

    loss, video_loss, token_loss, frame_out, token_out = build(params,
                                                               frame_input,
                                                               cat_mask_src,
                                                               cat_mask_tgt,
                                                               token_x_input,
                                                               token_y_input,
                                                               frame_mask_src,
                                                               frame_mask_tgt,
                                                               token_mask)

    update_ops, learning_rate = get_optimizer(loss, params, manual_global_step)

    print('\n')
    param_count = int(sum(np.prod([d.size for d in variable.shape.dims]) for variable in graph.trainable_variables))
    var_count = int(sum(np.prod([d.size for d in variable.shape.dims]) for variable in graph.all_variables))

    constant = '  variables: '
    variable_mapping = [('Model', param_count - params.embedding_param_count),
                        ('Embedding', params.embedding_param_count),
                        ('Untrainable', var_count - param_count),
                        ('', 0),
                        ('Total trainable', param_count),
                        ('Total', var_count)]
    variable_mapping = [(name, f'{int(count):,}') for name, count in variable_mapping]
    max_str = max(len(name) for name, _ in variable_mapping)
    max_int = max(len(count) for _, count in variable_mapping)
    for name, count in variable_mapping:
        if not name:
            print('-' * (max_str + max_int + len(constant)))
            continue
        print(f'{name:<{max_str}s}{constant}{count:>{max_int}s}')

    print("\nDimensions:")
    for dim_name in sorted(list(set([item for variable in graph.all_variables
                                     for item in variable.shape.dimension_names]))):
        print(dim_name)
    print('\n')

    lowering = mtf.Lowering(graph, {params.mesh: params.mesh_impl})

    mtf.scalar_summary("loss", loss)
    host_call = create_host_call(params.model_path)
    mtf.utils.remove_summaries()

    # Creates train_op
    global_step = tf.train.get_or_create_global_step()

    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    step = tf.mod(manual_global_step, tf.constant(params.grad_accumulation, dtype=tf.int64))
    step = tf.equal(step, tf.constant(0, dtype=tf.int64))
    step = tf.cast(step, tf.int64)
    tf_update_ops.append(tf.assign_add(global_step, step))
    tf_update_ops.append(tf.assign_add(manual_global_step, tf.constant(1, dtype=tf.int64, shape=[])))

    train_op = tf.group(tf_update_ops)

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.cast(tf_loss, tf.float32)
    with mtf.utils.outside_all_rewrites():
        hooks = [mtf.MtfRestoreHook(lowering)]
        if params.use_checkpointing:
            saver = tf.add_to_collection(tf.GraphKeys.SAVERS, tf.train.Saver(tf.global_variables(),
                                                                             sharded=True,
                                                                             max_to_keep=1,
                                                                             defer_build=False,
                                                                             save_relative_paths=True))
            hooks.append(tf.train.CheckpointSaverHook(params.model_path,
                                                      save_steps=params.steps_per_checkpoint,
                                                      saver=saver,
                                                      listeners=[mtf.MtfCheckpointSaverListener(lowering)]))

        return tpu_estimator.TPUEstimatorSpec(tf.estimator.ModeKeys.TRAIN,
                                              loss=tf_loss,
                                              host_call=host_call,
                                              training_hooks=hooks,
                                              prediction_hooks=[mtf.MtfRestoreHook(lowering)],
                                              train_op=train_op)
