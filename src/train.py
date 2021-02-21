"""
Contains functions to create a training loop and log its outputs to tensorboard
"""
import threading
import time
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.tpu import tpu, tpu_feed
from tensorflow.python.tpu.ops import tpu_ops

from .dataclass import ModelParameter
from .model import build
from .optimizers import get_optimizer
from .utils_mtf import pad, to_float, weighted_add


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


def masked_add(left, right, position):
    return weighted_add(left, right, mtf.one_hot(position, s))


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


def _import_tensor(params, tensor, shape, name):
    return mtf.import_laid_out_tensor(params.mesh, params.mesh_impl.LaidOutTensor([tensor]), shape, name)


def computation_func(params: ModelParameter, input_fn: typing.Callable,
                     session_config, cluster_resolver, callback_fns):
    # TODO(Lucas): move tf dataset to iterator/queue
    # TODO(Lucas): clean up code + optimize
    host_id_to_tf_device = "/job:worker/task:{:d}/device:CPU:0"
    hooks = []
    output_shapes = []
    tf.config.optimizer.set_experimental_options(params.tensorflow_optimization_settings)

    def _model_fn(*args):
        # Construct mtf graph + mesh from params
        graph = mtf.Graph()

        # Build mtf mesh object
        params.mesh = mtf.Mesh(graph, "mesh", mtf.utils.BalancedVariablePlacer(params.cpu_devices))

        # Build mtf_features & seq length dict for getting number of microbatches
        # We need to pack inputs into a dict to pass into serialize_training_step
        # params.mode = mode

        frame_input = None
        token_x_input = None
        token_y_input = None
        frame_mask = None
        token_mask = None

        if params.use_video:
            frame_input = _import_tensor(params, args[0], params.frame_input_shape, "frame_input")

            if params.use_language:
                token_x_input = _import_tensor(params, args[1], params.token_dim_shape, "tkn_src")
                token_y_input = _import_tensor(params, args[2], params.token_dim_shape, "tkn_tgt")
                frame_mask = _import_tensor(params, args[3], params.frame_mask_shape, "vid_msk")
                token_mask = _import_tensor(params, args[4], params.token_dim_shape, "txt_msk")

        else:  # params.use_language
            token_x_input = _import_tensor(params, args[0], params.token_dim_shape, "tkn_src")
            token_y_input = _import_tensor(params, args[1], params.token_dim_shape, "tkn_tgt")

        if params.train or not params.use_autoregressive_sampling:
            loss, video_loss, token_loss, frame_out, token_out = build(params,
                                                                       frame_input,
                                                                       token_x_input,
                                                                       token_y_input,
                                                                       frame_mask,
                                                                       token_mask)
        else:
            if params.use_video:
                tkn_per_frame = mtf.Dimension("language_token_per_frame",
                                              params.language_token_per_frame)
                shape = [params.batch_dim, params.sequence_dim, tkn_per_frame, params.vocab_dim]
                steps = params.time_patch_size

                def body_fn(position, video_loss, token_x_input, token_y_input, frame_input, frame_mask, token_mask,
                            *states):
                    _, video_loss, _, frame_out, token_out = build(params,
                                                                   frame_input,
                                                                   token_x_input,
                                                                   token_y_input,
                                                                   frame_mask,
                                                                   token_mask)

                    frame_input = weighted_add(pad(frame_out, params.sequence_dim, (0, 1)), frame_input,
                                               mtf.one_hot(position, params.frame_input_sequence, dtype=tf.float32))

                    if params.use_language:
                        one_hot_sequence = mtf.one_hot(position, params.sequence_dim, dtype=tf.float32)
                        token_out = mtf.argmax(mtf.reshape(token_out, new_shape=shape), reduced_dim=params.vocab_dim)
                        padding_token = to_float(mtf.equal(token_out, params.padding_token))

                        token_x_input = weighted_add(mtf.reshape(token_out, new_shape=params.token_dim_shape),
                                                     token_x_input,
                                                     mtf.one_hot(position, params.sequence_dim, dtype=tf.int32))

                        token_pad = mtf.less_equal(mtf.range(params.mesh, tkn_per_frame, dtype=tf.float32),
                                                   to_float(mtf.argmax(padding_token, reduced_dim=tkn_per_frame)),
                                                   output_shape=token_out.shape)

                        token_mask = weighted_add(mtf.reshape(to_float(token_pad), new_shape=params.token_dim_shape),
                                                  to_float(token_mask), one_hot_sequence)

                        frame_pad = to_float(mtf.equal(mtf.reduce_sum(padding_token, reduced_dim=tkn_per_frame), 0))
                        frame_mask = weighted_add(frame_pad, to_float(frame_mask), one_hot_sequence)

                    return position + 1, video_loss, token_x_input, token_y_input, frame_input, frame_mask, token_mask

                if token_mask is not None:
                    token_mask = to_float(token_mask)
                if frame_mask is not None:
                    frame_mask = to_float(frame_mask)

                while_loop_inputs = [mtf.zeros(params.mesh, [], tf.int32) + params.initial_autoregressive_position,
                                     mtf.zeros(params.mesh, [], params.variable_dtype.activation_dtype),
                                     token_x_input, token_y_input, frame_input, frame_mask, token_mask]
            else:  # -> params.use_language
                steps = params.sequence_dim.size

                def body_fn(position, token_loss, token_x, token_y, *states):
                    _, _, token_loss, _, token_out = build(params,
                                                           mtf.ones(params.mesh, [], tf.float32),
                                                           token_x,
                                                           token_y,
                                                           mtf.ones(params.mesh, [], tf.float32),
                                                           mtf.ones(params.mesh, [], tf.float32))
                    return (position + 1,
                            token_loss,
                            weighted_add(mtf.argmax(token_out, reduced_dim=params.vocab_dim), token_x,
                                         mtf.one_hot(position, output_dim=params.sequence_dim, dtype=tf.int32)),
                            token_y)

                while_loop_inputs = [mtf.zeros(params.mesh, [], tf.int32) + params.initial_autoregressive_position,
                                     mtf.zeros(params.mesh, [], params.variable_dtype.activation_dtype),
                                     token_x_input, token_y_input]

            def cond_fn(position, *states):
                is_done = mtf.greater_equal(position, steps)
                is_done = mtf.logical_or(is_done,
                                         mtf.greater_equal(position - params.initial_autoregressive_position, steps))
                is_done = mtf.reduce_sum(is_done)

                return mtf.logical_not(is_done)

            loop_out = mtf.while_loop(cond_fn=cond_fn, body_fn=body_fn, inputs=while_loop_inputs)

            if params.use_language:
                token_out = loop_out[2]
            if params.use_video:
                frame_out = loop_out[4]

        if params.train:
            update_ops = get_optimizer(loss, params)
        else:
            if params.use_language:
                token_out = mtf.anonymize(token_out)
            if params.use_video:
                frame_out = mtf.anonymize(frame_out)

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

        lowering = mtf.Lowering(graph, {params.mesh: params.mesh_impl}, autostack=True)

        if params.train:
            log_dict = {}
            if params.use_video:
                log_dict['video_loss'] = tf.cast(lowering.export_to_tf_tensor(video_loss), tf.float32)

            if params.use_language:
                log_dict['token_loss'] = tf.cast(lowering.export_to_tf_tensor(token_loss), tf.float32)
            write_summary = [add_summary(tf_loss=tf.cast(lowering.export_to_tf_tensor(loss), tf.float32),
                                         value=log_dict,
                                         global_step=tf.train.get_or_create_global_step()),
                             tf.assign_add(tf.train.get_or_create_global_step(), 1)
                             ] + [lowering.lowered_operation(op) for op in update_ops]
            with mtf.utils.outside_all_rewrites():
                hooks.append(mtf.MtfRestoreHook(lowering))
                if params.use_checkpointing:
                    saver = tf.train.Saver(tf.global_variables(),
                                           sharded=True,
                                           max_to_keep=1,
                                           keep_checkpoint_every_n_hours=2,
                                           defer_build=False,
                                           save_relative_paths=True)
                    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
                    hooks.append(tf.train.CheckpointSaverHook(params.model_path,
                                                              save_steps=params.steps_per_checkpoint,
                                                              saver=saver,
                                                              listeners=[mtf.MtfCheckpointSaverListener(lowering)]))

                return tf.group(write_summary)
        else:  # train == 'sample'
            predictions = {}
            if params.use_video:
                predictions['frame_out'] = lowering.export_to_tf_tensor(frame_out)
                predictions['frame_tgt'] = args[0]
            if params.use_language:
                predictions['token_out'] = lowering.export_to_tf_tensor(token_out)
                predictions['token_tgt'] = args[1 + int(params.use_video)]
            predictions = [val if val.dtype == tf.float32 else tf.cast(val, tf.float32) for val in predictions.values()]
            output_shapes.extend([pred.shape for pred in predictions])
            hooks.append(mtf.MtfRestoreHook(lowering))
            return tpu_ops.outfeed_enqueue_tuple(predictions)

    input_initializers = []
    enqueue = None

    num_cores = params.mesh_impl.device_assignment.num_replicas

    ordered_ordinals = np.zeros((num_cores,), dtype=np.int32)
    ordered_hosts = np.zeros((num_cores,), dtype=str)
    ordered_host_ids = np.zeros((num_cores,), dtype=np.int32)

    for pnum in range(num_cores):
        physical_pnum = params.mesh_impl.l2p(pnum)
        host_device = params.mesh_impl.device_assignment.host_device(replica=physical_pnum)
        # For MTF, there's always 1 core per replica. So logical_core=0.
        ordered_ordinals[pnum] = params.mesh_impl.device_assignment.tpu_ordinal(replica=physical_pnum, logical_core=0)
        ordered_hosts[pnum] = host_device
        ordered_host_ids[pnum] = int(host_device.lower().split("/task:")[1].split("/device:")[0])

    num_hosts = len(set(ordered_host_ids))
    pnum_maps = []
    batch_size = params.input_pipeline_shape[0].to_integer_list[0]

    for shape in params.input_pipeline_shape:
        # Make sure that the batch size is the same across all input tensors.
        if batch_size != shape.to_integer_list[0]:
            raise ValueError
        s_shape = params.mesh_impl.slice_shape(shape)
        shape_list = [dim_size // s_dim_size for dim_size, s_dim_size in zip(shape.to_integer_list, s_shape)]
        pnum_map = -np.ones(shape_list + [num_cores // np.prod(shape_list)], dtype=np.int32)

        for pnum in range(num_cores):
            coord = tuple([d // s for d, s in zip(params.mesh_impl.slice_begin(shape, pnum), s_shape)])
            pnum_array_ref = pnum_map[coord]
            for idx, value in enumerate(pnum_array_ref):
                if value == -1:
                    pnum_array_ref[idx] = pnum
                    break

        if np.any(pnum_map == -1):
            raise ValueError
        pnum_maps.append(pnum_map)

    # For each sub-batch, we need to know which host should read it.
    hosts_to_hold_ds = [num_hosts - 1]
    if params.train:
        hosts_to_hold_ds.clear()
        num_dss_per_host = np.zeros((num_hosts,))

        for sub_batch_pnum_map in pnum_maps[0]:
            host_id = np.argmax(np.sum(np.equal(ordered_host_ids.take(sub_batch_pnum_map.flatten(), 0).reshape(1, -1),
                                                np.arange(num_hosts).reshape(-1, 1)), 1) + num_dss_per_host)
            num_dss_per_host[host_id] -= 0.1 / num_hosts
            hosts_to_hold_ds.append(host_id)

    sub_batch_size = batch_size // len(hosts_to_hold_ds)

    if sub_batch_size * len(hosts_to_hold_ds) != batch_size:
        raise ValueError

    # For each sub-batch, create a SubBatchSlicer object.
    # Get the list of pnums for each input.
    all_laidout_tensors = [[None] * len(params.input_pipeline_shape) for _ in range(num_cores)]
    for sub_batch_i, host_id in enumerate(hosts_to_hold_ds):
        with ops.device(host_id_to_tf_device.format(host_id)):
            ds_iterator = input_fn(params, sub_batch_size, sub_batch_i,
                                   len(hosts_to_hold_ds)).make_initializable_iterator()
            input_initializers.append(ds_iterator.initializer)

            all_input_tensors = ds_iterator.get_next()
            all_sub_batch_pnums = [pnum_map.flatten().tolist() if not params.train else
                                   pnum_map[sub_batch_i, ...].flatten().tolist()
                                   for pnum_map in pnum_maps]
            if len(all_input_tensors) != len(params.input_pipeline_shape):
                raise ValueError

            for idx, input_tensor in enumerate(all_input_tensors):
                sub_batch_pnums = all_sub_batch_pnums[idx]
                mtf_input_shape = params.input_pipeline_shape[idx]

                # Initialize the cache for each
                slice_dict = {}

                for pnum in sub_batch_pnums:
                    s_begin = params.mesh_impl.slice_begin(mtf_input_shape, pnum)
                    s_begin[0] = s_begin[0] % sub_batch_size * (not params.train)
                    s_begin = tuple(s_begin)
                    if s_begin in slice_dict:
                        all_laidout_tensors[pnum][idx] = tf_tensor
                        continue
                    tf_tensor = tf.slice(input_tensor, s_begin, params.mesh_impl.slice_shape(mtf_input_shape))

                    slice_dict[s_begin] = tf_tensor
                    all_laidout_tensors[pnum][idx] = tf_tensor

    with ops.device(host_id_to_tf_device.format(hosts_to_hold_ds[0])):
        laidout_tensors0 = all_laidout_tensors[0]
        infeed = tpu_feed.InfeedQueue(number_of_tuple_elements=len(laidout_tensors0),
                                      tuple_types=[x.dtype for x in laidout_tensors0],
                                      tuple_shapes=[x.shape for x in laidout_tensors0])
        enqueue = infeed.generate_enqueue_ops(all_laidout_tensors,
                                              tpu_ordinal_function=lambda x: ordered_ordinals[x],
                                              placement_function=lambda x: ordered_hosts[x])

    def _thread_fn(sess: tf.Session):
        time.sleep(1)
        while True:
            sess.run(enqueue)

    compilation_state, computation = tpu.split_compile_and_replicate(_model_fn,
                                                                     [[]] * params.num_cores,
                                                                     infeed,
                                                                     params.d_assignment,
                                                                     None,
                                                                     maximum_shapes=None)
    ckpt_loader_hook = CheckpointLoaderHook(params.model_path)

    if params.train:
        # if params.write_summary:
        flush_summary = summary.flush()

        with tf.train.MonitoredTrainingSession(master=cluster_resolver.master(),
                                               hooks=[ckpt_loader_hook,
                                                      tf.train.StepCounterHook(every_n_steps=10)] + hooks,
                                               config=session_config) as sess:
            sess.run(input_initializers)
            infeed_thread = threading.Thread(target=_thread_fn, args=(sess,))
            infeed_thread.start()
            summary.initialize(session=sess)

            for i in range(params.current_step, params.train_steps):
                sess.run(computation)
                if (i + 1) % params.summary_flush_interval == 0:
                    sess.run(flush_summary)
                for fn in callback_fns:
                    fn(i)

    else:  # train == 'sample'
        outfeed_dequeue_ops = []
        for host_id in range(params.num_hosts):
            with ops.device(host_id_to_tf_device.format(host_id)):
                for device_ordinal in range(params.num_cores_per_host):
                    outfeed_dequeue_op = tpu_ops.outfeed_dequeue_tuple(dtypes=[tf.float32] * len(output_shapes),
                                                                       shapes=output_shapes,
                                                                       device_ordinal=device_ordinal)
                    # We don't need output other than from core 0.
                    outfeed_dequeue_ops.append([tf.reduce_mean(x) for x in outfeed_dequeue_op]
                                               if outfeed_dequeue_ops else outfeed_dequeue_op)
        with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(master=cluster_resolver.master(),
                                                                                    config=session_config),
                                       hooks=[ckpt_loader_hook, hooks[0]]) as sess:
            sess.run(input_initializers)
            # error probably here -> it didnt run init
            infeed_thread = threading.Thread(target=_thread_fn, args=(sess,))
            infeed_thread.start()
            while True:
                sess.run(computation)
                out = sess.run(outfeed_dequeue_ops)[0]
                for fn in callback_fns:
                    fn(out)
