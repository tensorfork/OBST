"""
Contains functions to create a training loop and log its outputs to tensorboard
"""
import collections
import json
import time
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2 as summary, variables
from tensorflow.python.tpu import tpu, tpu_feed
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import checkpoint_management

from .dataclass import ModelParameter
from .model import build
from .mtf_wrapper import constant_scalar, log
from .optimizers import get_optimizer
from .utils_core import color_print
from .utils_mtf import concat, pad, slice, to_fp32, weighted_add, anonymize, head_argmax

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


def computation_func(params: ModelParameter, input_fn: typing.Callable,
                     session_config, cluster_resolver, callback_fns, query_input_fns=None):
    # TODO(Lucas): move tf dataset to iterator/queue
    # TODO(Lucas): clean up code + optimize
    host_id_to_tf_device = "/job:worker/task:{:d}/device:CPU:0"
    hooks = []
    output_shapes = []
    tf.config.optimizer.set_experimental_options(params.tensorflow_optimization_settings)

    def _model_fn(*args):
        manual_global_step = tf1.get_variable("manual_global_step", [], tf.int64, initializer=tf.zeros_initializer(),
                                              trainable=False,
                                              aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)
        # Construct mtf graph + mesh from params
        graph = mtf.Graph()

        # Build mtf mesh object
        params.mesh = mtf.Mesh(graph, "mesh", mtf.utils.BalancedVariablePlacer(params.cpu_devices))

        def _base_model_fn(*args):

            if params.macro_batching > 1 and params.train:
                loop_idx, prev_loss, *args = args
                args = list(args)[log_len:]
                inp_args = args.copy()

                for _inp_idx, _inp in enumerate(args):
                    slice_shape = [loop_idx] + [0 for _ in range(len(_inp.shape) - 1)]
                    size_shape = list(_inp.shape)
                    size_shape[0] = params.train_batch_size // params.batch_splits

                    args[_inp_idx] = tf.slice(_inp, slice_shape, size_shape)

            # Build mtf_features & seq length dict for getting number of microbatches
            # We need to pack inputs into a dict to pass into serialize_training_step
            # params.mode = mode

            frame_input = None
            cat_mask_src = None
            cat_mask_tag = None
            token_x_input = None
            token_y_input = None
            frame_mask_src = None
            frame_mask_tag = None
            token_mask = None
            initial_pos = None
            sampling_temperature = None
            end_iterations = None

            start_time = time.time()
            color_print(params, "Building Mesh-TensorFlow graph...")
            if params.use_video:
                frame_input = _import_tensor(params, args[0], params.frame_input_shape, "frame_input")
                cat_mask_src = _import_tensor(params, args[1], params.frame_mask_shape, "cat_mask_x")
                cat_mask_tag = _import_tensor(params, args[2], params.frame_mask_shape, "cat_mask_y")
                frame_mask_src = _import_tensor(params, args[3], params.frame_mask_shape, "vid_msk_src")
                frame_mask_tag = _import_tensor(params, args[4], params.frame_mask_shape, "vid_msk_tgt")

                if params.use_language:
                    token_x_input = _import_tensor(params, args[5], params.token_dim_shape, "tkn_src")
                    token_y_input = _import_tensor(params, args[6], params.token_dim_shape, "tkn_tgt")
                    token_mask = _import_tensor(params, args[7], params.token_dim_shape, "txt_msk")

            else:  # params.use_language
                token_x_input = _import_tensor(params, args[0], params.token_dim_shape, "tkn_src")
                token_y_input = _import_tensor(params, args[1], params.token_dim_shape, "tkn_tgt")

                if not query_input_fns is None:
                    initial_pos_dim = mtf.Dimension("_initial_pos_dim", 1)
                    initial_pos = _import_tensor(params, args[2], mtf.Shape([initial_pos_dim]), "initial_pos")
                    initial_pos = mtf.reduce_sum(initial_pos, output_shape=[])
                    sampling_temperature = _import_tensor(params, args[3], mtf.Shape([initial_pos_dim]), "temperature")
                    sampling_temperature = mtf.reduce_sum(sampling_temperature, output_shape=[])
                    end_iterations = _import_tensor(params, args[4], mtf.Shape([initial_pos_dim]), "end_iterations")
                    end_iterations = mtf.reduce_sum(end_iterations, output_shape=[])

            if params.train or not params.use_autoregressive_sampling:
                loss, loss_list, video_loss, accuracy, token_loss, frame_out, token_out = build(params,
                                                                                                frame_input,
                                                                                                cat_mask_src,
                                                                                                cat_mask_tag,
                                                                                                token_x_input,
                                                                                                token_y_input,
                                                                                                frame_mask_src,
                                                                                                frame_mask_tag,
                                                                                                token_mask)
            else:
                if params.use_video:
                    # todo: fix token shift for video (Jan).
                    tkn_per_frame = mtf.Dimension("language_token_per_frame",
                                                  params.language_token_per_frame)
                    shape = [params.batch_dim, params.sequence_dim, tkn_per_frame] + params.vocab_dims
                    steps = params.time_patch_size

                    def body_fn(position, token_x_input, token_y_input, frame_input,
                                frame_mask_src, frame_mask_tag, token_mask, *states):

                        _, _, _, _, _, frame_out, token_out = build(params,
                                                                    frame_input,
                                                                    mtf.ones(params.mesh, [], tf.float32),
                                                                    mtf.ones(params.mesh, [], tf.float32),
                                                                    token_x_input,
                                                                    token_y_input,
                                                                    frame_mask_src,
                                                                    frame_mask_tag,
                                                                    token_mask)

                        frame_input = weighted_add(pad(frame_out, params.sequence_dim, (0, 1)), frame_input,
                                                   mtf.one_hot(position, params.frame_input_sequence, dtype=tf.float32))

                        if params.use_language:
                            one_hot_sequence = mtf.one_hot(position, params.sequence_dim, dtype=tf.float32)
                            token_out = head_argmax(mtf.reshape(token_out, new_shape=shape), params.vocab_dims)
                            padding_token = to_fp32(mtf.equal(token_out, params.padding_token))

                            token_x_input = weighted_add(mtf.reshape(token_out, new_shape=params.token_dim_shape),
                                                         token_x_input,
                                                         mtf.one_hot(position, params.sequence_dim, dtype=tf.int32))

                            token_pad = mtf.less_equal(mtf.range(params.mesh, tkn_per_frame, dtype=tf.float32),
                                                       to_fp32(mtf.argmax(padding_token, reduced_dim=tkn_per_frame)),
                                                       output_shape=token_out.shape)

                            token_mask = weighted_add(
                                    mtf.reshape(to_fp32(token_pad), new_shape=params.token_dim_shape),
                                    to_fp32(token_mask), one_hot_sequence)

                            frame_pad = to_fp32(
                                    mtf.greater(mtf.reduce_sum(padding_token, reduced_dim=tkn_per_frame), 0))
                            token_x_input = weighted_add(frame_pad, to_fp32(token_x_input), one_hot_sequence)

                            token_x_input = mtf.cast(token_x_input, dtype=tf.int32)

                        return position + 1, token_x_input, token_y_input, frame_input, frame_mask_src, \
                               frame_mask_tag, token_mask

                    if token_mask is not None:
                        token_mask = to_fp32(token_mask)
                    if frame_mask_src is not None:
                        frame_mask_src = to_fp32(frame_mask_src)
                    if frame_mask_tag is not None:
                        frame_mask_tag = to_fp32(frame_mask_tag)

                    while_loop_inputs = [mtf.zeros(params.mesh, [], tf.int32) + params.initial_autoregressive_position,
                                         token_x_input, token_y_input, frame_input, frame_mask_src, frame_mask_tag,
                                         token_mask]

                else:  # -> params.use_language
                    def body_fn(position, token_x, token_y, sampling_temperature, *states):
                        _, _, _, _, _, _, token_out = build(params,
                                                            mtf.ones(params.mesh, [], tf.float32),
                                                            mtf.ones(params.mesh, [], tf.float32),
                                                            mtf.ones(params.mesh, [], tf.float32),
                                                            token_x,
                                                            token_y,
                                                            mtf.ones(params.mesh, [], tf.float32),
                                                            mtf.ones(params.mesh, [], tf.float32),
                                                            mtf.ones(params.mesh, [], tf.float32))

                        one_hot_mask = mtf.one_hot(position, output_dim=params.sequence_dim, dtype=tf.int32)

                        token_out += (log(-log(mtf.random_uniform(params.mesh, token_out.shape, maxval=1,
                                                                      minval=1e-9, dtype=tf.float32))))
                        token_out = mtf.argmax(token_out, params.vocab_dim)

                        token_out = mtf.shift(token_out, offset=1, dim=params.sequence_dim, wrap=False)

                        return (position + 1, weighted_add(token_out, token_x, one_hot_mask),
                                token_y, mtf.cast(sampling_temperature, dtype=tf.float32))

                    if initial_pos is None:
                        initial_pos = mtf.constant(params.mesh, value=params.initial_autoregressive_position,
                                                   dtype=tf.int32)
                    token_initial_pos_mask = mtf.less_equal(mtf.range(params.mesh, params.sequence_dim, dtype=tf.int32),
                                                            initial_pos)
                    token_initial_pos_mask = mtf.cast(token_initial_pos_mask, tf.int32)

                    if params.debug_sample:
                        token_x_input_a = slice(token_x_input, 0, 1, dim=params.batch_dim)
                        token_x_input_b = token_x_input_a * token_initial_pos_mask
                        token_x_input = concat([token_x_input_a, token_x_input_b], dim=token_x_input_a.shape[0])
                    else:
                        token_x_input = token_x_input * token_initial_pos_mask

                    if sampling_temperature is None:
                        sampling_temperature = constant_scalar(params, params.sampling_temperature, dtype=tf.float32)

                    if end_iterations is None:
                        end_iterations = mtf.constant(params.mesh, value=params.n_ctx,
                                                   dtype=tf.int32)

                    while_loop_inputs = [initial_pos, token_x_input, token_y_input, sampling_temperature]

                def cond_fn(position, *states):
                    is_done = mtf.greater_equal(position, end_iterations)
                    is_done = mtf.reduce_sum(is_done)

                    return mtf.logical_not(is_done)

                loop_out = mtf.while_loop(cond_fn=cond_fn, body_fn=body_fn, inputs=while_loop_inputs)

                if params.use_language:
                    token_out = loop_out[1]
                if params.use_video:
                    frame_out = loop_out[3]

            if params.train:
                if params.multi_loss_strategy == "linear":
                    loss_list = [loss]
                elif params.multi_loss_strategy == "mgda":
                    loss_list = loss_list + [None]

                update_ops, learning_rate, debug_gradients_dict = get_optimizer(loss_list, params, manual_global_step)
            else:
                if params.use_language:
                    token_out = mtf.anonymize(token_out)
                if params.use_video:
                    if params.use_discrete_video_loss:
                        frame_out = mtf.argmax(frame_out, reduced_dim=params.discrete_color_dim)
                    frame_out = mtf.anonymize(frame_out)

            color_print(params, f"Built in {time.time() - start_time:.1f}s")
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

            model_size = {'model_variables':           int(param_count - embed_param_count),
                          'embedding_variables':       int(embed_param_count),
                          'body_variables':            int(body_param_count),
                          'untrainable_variables':     int(var_count - param_count),
                          'total_trainable_variables': int(param_count),
                          'total_variables':           int(var_count)
                          }

            if params.train:
                json.dump(model_size, tf.io.gfile.GFile(f"{params.model_path}/model_size.info", 'w'), indent=2)

            color_print(params, "Lowering graph to TensorFlow...")
            start_time = time.time()
            lowering = mtf.Lowering(graph, {params.mesh: params.mesh_impl})
            color_print(params, f"Lowered in {time.time() - start_time:.1f}s")
            if params.train:
                log_dict = {'learning_rate': tf.cast(learning_rate, tf.float32)}
                if params.use_video:
                    log_dict['video_loss'] = tf.cast(lowering.export_to_tf_tensor(video_loss), tf.float32)
                if params.use_language:
                    log_dict['token_loss'] = tf.cast(lowering.export_to_tf_tensor(token_loss), tf.float32)
                if accuracy is not None:
                    log_dict['accuracy'] = tf.cast(lowering.export_to_tf_tensor(accuracy), tf.float32)

                comput_ops = [lowering.lowered_operation(op) for op in update_ops]

                with tf.control_dependencies(comput_ops):
                    global_step = tf1.train.get_or_create_global_step()

                    step = tf.math.mod(manual_global_step + 1, tf.constant(params.grad_accumulation, dtype=tf.int64))
                    step = tf.equal(step, tf.constant(0, dtype=tf.int64))
                    step = tf.cast(step, tf.int64)

                    tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)

                    if params.macro_batching > 1 and params.train:
                        if params.macro_batch_loss_smoothing:
                            tf_loss = (tf_loss + prev_loss * loop_idx) / (1 + loop_idx)
                        params.log_dict_keys = list(log_dict.keys())
                    else:
                        comput_ops.append(add_summary(tf_loss=tf_loss, value=log_dict, global_step=global_step))

                    if params.debug_gradients:
                        for grad_key in debug_gradients_dict.keys():
                            debug_gradients_dict[grad_key] = \
                                tf.cast(lowering.export_to_tf_tensor(debug_gradients_dict[grad_key]), tf.float32)

                        comput_ops.append(add_histogram(tf_loss=tf_loss, value=debug_gradients_dict,
                                                        global_step=global_step))

                    comput_ops.extend([tf1.assign_add(global_step, step),
                                       tf1.assign_add(manual_global_step, tf.constant(1, dtype=tf.int64, shape=[]))])

                hooks.append(mtf.MtfRestoreHook(lowering))
                with mtf.utils.outside_all_rewrites():
                    if params.use_checkpointing:
                        saver = tf1.train.Saver(tf1.global_variables(),
                                                sharded=True,
                                                max_to_keep=params.max_checkpoints_keep,
                                                defer_build=False,
                                                save_relative_paths=True)
                        tf1.add_to_collection(tf1.GraphKeys.SAVERS, saver)
                        hooks.append(tf1.train.CheckpointSaverHook(params.model_path,
                                                                   save_steps=params.steps_per_checkpoint,
                                                                   saver=saver,
                                                                   listeners=[mtf.MtfCheckpointSaverListener(lowering)],
                                                                   save_graph_def=params.save_graph))
                        ckpt = checkpoint_management.get_checkpoint_state(params.model_path)
                        if ckpt is not None:
                            color_print(params, "Recovering last checkpoints...")
                            saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

                if params.macro_batching > 1 and params.train:
                    with tf.control_dependencies(comput_ops):
                        return [loop_idx + 1, tf_loss] + [log_dict[key] for key in params.log_dict_keys] + inp_args
                else:
                    return tf.group(comput_ops)

            else:  # train == 'sample'
                predictions = {}

                if params.use_video:
                    predictions['frame_out'] = lowering.export_to_tf_tensor(frame_out)
                    predictions['frame_tgt'] = args[0]

                if params.use_language:
                    predictions['token_out'] = lowering.export_to_tf_tensor(token_out)
                    predictions['token_tgt'] = args[1 + int(params.use_video) * 5]

                predictions = [val if val.dtype == tf.float32 else tf.cast(val, tf.float32) for val in
                               predictions.values()]
                output_shapes.extend([pred.shape for pred in predictions])
                hooks.append(mtf.MtfRestoreHook(lowering))
                return tpu_ops.outfeed_enqueue_tuple(predictions)

        if params.train and params.macro_batching > 1:
            log_len = int(params.use_language) + int(params.use_video) + \
                      int(params.calc_accuracy) * int(params.use_language) + 1
            loop_inputs = [tf.constant(0, dtype=tf.int32, shape=[]), tf.constant(0, dtype=tf.float32, shape=[])]
            loop_inputs = loop_inputs + [tf.constant(0, dtype=tf.float32, shape=[]) for _ in range(log_len)] \
                          + list(args)

            def con(i, *args):
                return tf.less(i, tf.constant(params.macro_batching, dtype=tf.int32, shape=[]))

            loop_out = tf.while_loop(cond=con, body=_base_model_fn,
                                     loop_vars=loop_inputs, back_prop=False, parallel_iterations=1)

            tf_loss = loop_out[1]
            log_list = loop_out[2:][:log_len]
            log_dict = {key: val for (key, val) in zip(params.log_dict_keys, log_list)}
            global_step = tf1.train.get_or_create_global_step()

            with tf.control_dependencies(log_list):
                ret = add_summary(tf_loss=tf_loss, value=log_dict, global_step=global_step)
        else:
            ret = _base_model_fn(*args)

        return ret

    num_cores = params.mesh_impl.device_assignment.num_replicas

    ordered_ordinals = []
    ordered_hosts = []
    ordered_host_ids = []
    host_id_to_its_pnums = collections.defaultdict(list)
    d_assignment = params.mesh_impl.device_assignment

    for pnum in range(num_cores):
        physical_pnum = params.mesh_impl.l2p(pnum)

        # For MTF, there's always 1 core per replica. So logical_core=0.
        ordered_ordinals.append(d_assignment.tpu_ordinal(replica=physical_pnum, logical_core=0))
        host_device = d_assignment.host_device(replica=physical_pnum)
        host_id = int(host_device.lower().split("/task:")[1].split("/device:")[0])
        ordered_hosts.append(host_device)
        ordered_host_ids.append(host_id)
        host_id_to_its_pnums[host_id].append(pnum)

    if query_input_fns is None:
        num_hosts = len(set(ordered_hosts))

        pnum_maps = []
        macro_batching_multi = params.macro_batching if params.train else 1
        batch_size = params.input_pipeline_shape[0].to_integer_list[0] * macro_batching_multi
        for mtf_shape in params.input_pipeline_shape:
            # Make sure that the batch size is the same across all input tensors.
            assert batch_size == mtf_shape.to_integer_list[0] * macro_batching_multi

            s_shape = params.mesh_impl.slice_shape(mtf_shape)
            shape_list = [dim_size // s_dim_size for dim_size, s_dim_size in zip(mtf_shape.to_integer_list, s_shape)]

            pnum_map_shape = shape_list + [num_cores // np.prod(shape_list)]
            assert np.prod(pnum_map_shape) == num_cores

            # Initialize the pnum_map to None.
            pnum_map = np.empty(pnum_map_shape, dtype=object)
            pnum_map[:] = None

            for pnum in range(num_cores):
                s_begin = params.mesh_impl.slice_begin(mtf_shape, pnum)
                coord = [dim_size // s_dim_size for dim_size, s_dim_size in zip(s_begin, s_shape)]
                # put pnum in pnum_map[coord]
                pnum_array_ref = pnum_map[tuple(coord)]
                for idx, value in enumerate(pnum_array_ref):
                    if value is None:
                        pnum_array_ref[idx] = pnum
                        break

            pnum_maps.append(pnum_map)

        # For each sub-batch, we need to know which host should read it.
        if params.train:

            # This records how many datasets (ds) are already stored on each host.
            num_dss_per_host = [0] * num_hosts

            # A list of host_ids that holds datasets (ds).
            hosts_to_hold_ds = []

            for sub_batch_pnum_map in pnum_maps[0]:

                num_pnums_per_host = [0] * num_hosts
                for pnum in sub_batch_pnum_map.flatten():
                    num_pnums_per_host[ordered_host_ids[pnum]] += 1

                host_metrics = [(host_id, num_pnums_per_host[host_id], num_dss_per_host[host_id]) for host_id in
                                range(num_hosts)]
                host_id, _, _ = max(host_metrics, key=lambda keys: (keys[1], -keys[2]))

                num_dss_per_host[host_id] += 1
                hosts_to_hold_ds.append(host_id)

        else:
            # There should be just one dataset-holding host. Make the last host do it.
            hosts_to_hold_ds = [num_hosts - 1]

        sub_batch_size = batch_size // len(hosts_to_hold_ds)
        tf1.logging.info("MTF sub_batch_size: {}".format(sub_batch_size))
        assert sub_batch_size * len(hosts_to_hold_ds) == batch_size

        # Slots for all laidout tensors.
        all_laidout_tensors = [[None] * len(params.input_pipeline_shape) for _ in range(num_cores)]

        log_path = params.model_path + "/DataLog.log"
        _run_log = []
        run_log = None

        if params.use_checkpointing:
            if tf.io.gfile.exists(log_path):
                _run_log = json.load(tf.io.gfile.GFile(log_path, 'r'))

            curran_stats = {'steps':             params.current_step, 'ctx': params.n_ctx,
                            'slice_count':       len(hosts_to_hold_ds),
                            'interleave_size':   params.interleaved_datasets,
                            'batch_size':        params.train_batch_size,
                            'grad_accumulation': params.grad_accumulation,
                            'token_patch_size':  params.token_patch_size
                            }
            json.dump((_run_log + [curran_stats]), tf.io.gfile.GFile(log_path, 'w'), indent=2)

            if len(_run_log) > 0 and not params.use_random_dataloader:
                _run_log = [r for r in _run_log if r['steps'] != params.current_step]
                if len(_run_log) > 0:
                    run_log = [_run_log.pop(-1)]
                    for r in _run_log[::-1]:
                        if run_log[-1]['steps'] != r['steps'] and r['steps'] != params.current_step:
                            run_log.append(r)
                    run_log = run_log[::-1]

                    for run_idx in range(len(run_log) - 1):
                        run_log[run_idx]['steps'] = run_log[run_idx + 1]['steps'] - run_log[run_idx]['steps']

                    run_log[-1]['steps'] = params.current_step - run_log[-1]['steps']

                    if run_log[-1]['steps'] <= 0:
                        run_log = None

        ds_iterator = []
        # For each sub-batch, create a SubBatchSlicer object.
        for sub_batch_i, host_id in enumerate(hosts_to_hold_ds):
            # Get the list of pnums for each input.
            if params.train:

                all_sub_batch_pnums = []
                for pnum_map in pnum_maps:
                    sub_batch_pnums = pnum_map[sub_batch_i, ...].flatten().tolist()
                    all_sub_batch_pnums.append(sub_batch_pnums)

            else:

                all_sub_batch_pnums = [pnum_map.flatten().tolist() for pnum_map in pnum_maps]

            with ops.device(f"/job:worker/task:{host_id}/device:CPU:0"):
                dataset = input_fn(params, sub_batch_size, sub_batch_i, len(hosts_to_hold_ds), run_log)
                if not params.use_random_dataloader and params.train and params.use_video:
                    dataset = dataset.skip(params.current_step // params.macro_batching)
                dataset = dataset.prefetch(params.buffer_size)
                options = tf.data.Options()
                options.experimental_deterministic = not params.train
                options.experimental_optimization.autotune = True
                options.experimental_optimization.autotune_buffers = True
                options.experimental_optimization.filter_fusion = True
                options.experimental_optimization.hoist_random_uniform = True
                options.experimental_optimization.map_and_batch_fusion = True
                options.experimental_optimization.map_and_filter_fusion = False
                options.experimental_optimization.map_fusion = True
                options.experimental_optimization.map_parallelization = True
                options.experimental_optimization.map_vectorization.enabled = True
                options.experimental_optimization.map_vectorization.use_choose_fastest = True
                options.experimental_optimization.noop_elimination = True
                options.experimental_optimization.parallel_batch = True
                options.experimental_optimization.shuffle_and_repeat_fusion = True
                options.experimental_optimization.apply_default_optimizations = False
                options.experimental_threading.max_intra_op_parallelism = 1
                options.experimental_threading.private_threadpool_size = 48
                options.experimental_distribute.auto_shard_policy = AutoShardPolicy.AUTO
                dataset: Dataset = dataset.with_options(options)
                _ds_iterator = tf1.data.make_initializable_iterator(dataset)
                ds_iterator.append(_ds_iterator)
                all_input_tensors = _ds_iterator.get_next()

                if isinstance(all_input_tensors, tf.Tensor):
                    all_input_tensors = [all_input_tensors]
                assert len(all_input_tensors) == len(all_sub_batch_pnums)

                for input_i in range(len(all_input_tensors)):
                    input_tensor = all_input_tensors[input_i]
                    sub_batch_pnums = all_sub_batch_pnums[input_i]
                    mtf_input_shape = params.input_pipeline_shape[input_i]

                    # Initialize the cache for each input_i
                    _slice_dict = collections.defaultdict(list)

                    for idx, pnum in enumerate(sub_batch_pnums):

                        s_begin = params.mesh_impl.slice_begin(mtf_input_shape, pnum)
                        if not not params.train:
                            # Always slice from 0 in the first dimension (batch dimension), since
                            # input_tensor a sub-batch tensor.
                            s_begin[0] = 0
                        if tuple(s_begin) in _slice_dict:
                            input_slice = _slice_dict[tuple(s_begin)]
                        else:
                            s_shape = params.mesh_impl.slice_shape(mtf_input_shape)
                            s_shape[0] = s_shape[0] * macro_batching_multi
                            input_slice = tf.slice(input_tensor, s_begin, s_shape)

                        all_laidout_tensors[pnum][input_i] = input_slice

        # Make sure that there are no Nones in all_laidout_tensors.
        for laidout_tensors in all_laidout_tensors:
            assert None not in laidout_tensors

        with ops.device(f"/job:worker/task:{hosts_to_hold_ds[0]}/device:CPU:0"):

            def _tpu_ordinal_function_impl(pnum):
                return ordered_ordinals[pnum]

            def _placement_function_impl(pnum):
                return ordered_hosts[pnum]

            laidout_tensors0 = all_laidout_tensors[0]
            infeed_queue = tpu_feed.InfeedQueue(
                    number_of_tuple_elements=len(laidout_tensors0),
                    tuple_types=[x.dtype for x in laidout_tensors0],
                    tuple_shapes=[x.shape for x in laidout_tensors0])
            enqueue_ops = infeed_queue.generate_enqueue_ops(
                    all_laidout_tensors,
                    tpu_ordinal_function=_tpu_ordinal_function_impl,
                    placement_function=_placement_function_impl)

        input_initializers = [ds.initializer for ds in ds_iterator]
    else:

        def _tpu_ordinal_function_impl(pnum):
            return ordered_ordinals[pnum]

        def _placement_function_impl(pnum):
            return ordered_hosts[pnum]

        prompt = tf1.placeholder(dtype=tf.int32, shape=[t.size for t in params.token_dim_shape])
        iter_pos = tf1.placeholder(dtype=tf.int32, shape=[1])
        samp_temp = tf1.placeholder(dtype=tf.float32, shape=[1])
        end_iter = tf1.placeholder(dtype=tf.int32, shape=[1])

        all_laidout_tensors = [[prompt, prompt, iter_pos, samp_temp, end_iter] for _ in range(params.num_cores)]

        laidout_tensors0 = all_laidout_tensors[0]
        infeed_queue = tpu_feed.InfeedQueue(
                number_of_tuple_elements=len(laidout_tensors0),
                tuple_types=[x.dtype for x in laidout_tensors0],
                tuple_shapes=[x.shape for x in laidout_tensors0])
        enqueue_ops = infeed_queue.generate_enqueue_ops(all_laidout_tensors,
                                                        tpu_ordinal_function=_tpu_ordinal_function_impl,
                                                        placement_function=_placement_function_impl)


    color_print(params, "Building split TensorFlow computation...")
    start_time = time.time()
    compilation_state, computation = tpu.split_compile_and_replicate(_model_fn,
                                                                     [[]] * params.num_cores,
                                                                     infeed_queue,
                                                                     params.d_assignment,
                                                                     None,
                                                                     maximum_shapes=None)
    color_print(params, f"Built computation in {time.time() - start_time:.1f}s")
    ckpt_loader_hook = CheckpointLoaderHook(params.model_path)
    color_print(params, "Connecting to TPU...")
    start_time = time.time()
    if params.train:
        # if params.write_summary:
        flush_summary = summary.flush()

        with tf1.train.MonitoredTrainingSession(master=cluster_resolver.master(),
                                                hooks=[ckpt_loader_hook,
                                                       tf1.train.StepCounterHook(every_n_steps=10)] + hooks,
                                                config=session_config) as sess:
            color_print(params, f"Connected after {time.time() - start_time:.1f}s")
            color_print(params, 'Compiling computation...')
            now = time.time()
            sess.run(compilation_state)
            elapsed = time.time() - now
            color_print(params, f'Compiled in {elapsed:.1f}s')

            color_print(params, "Initializing inputs...")
            sess.run(input_initializers)

            color_print(params, "Initializing summary...")
            summary.initialize(session=sess)

            now = time.time()
            color_print(params, f'Enqueueing first batch...')
            sess.run(enqueue_ops)
            elapsed = time.time() - now
            color_print(params, f'Enqueued in {elapsed:.1f}s')

            current_step = params.current_step
            color_print(params, f"Starting training loop. Start step: {current_step}")
            first_print_threshold = (5 + current_step) * params.grad_accumulation
            first_print_threshold *= np.maximum(1, (params.macro_batching // params.grad_accumulation))
            current_step = current_step * params.grad_accumulation
            for i in range(current_step, params.train_steps * params.grad_accumulation, params.macro_batching):

                sess.run(computation)
                if params.debug_train_step or i < first_print_threshold:
                    color_print(params, f"Current global step: {i // params.grad_accumulation}"
                                        f"   accumulation step: {i % params.grad_accumulation}")

                sess.run(enqueue_ops)
                if params.debug_train_step:
                    color_print(params, f"Enqueueing...")

                sess.run(flush_summary)
                if params.debug_train_step:
                    color_print(params, f"Flushing summary...")

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
        with tf1.train.MonitoredSession(session_creator=tf1.train.ChiefSessionCreator(master=cluster_resolver.master(),
                                                                                      config=session_config),
                                        hooks=[ckpt_loader_hook, hooks[0]]) as sess:

            color_print(params, f"Connected after {time.time() - start_time:.1f}s")
            color_print(params, 'Compiling computation...')
            now = time.time()
            sess.run(compilation_state)
            elapsed = time.time() - now
            color_print(params, f'Compiled in {elapsed:.1f}s')

            if query_input_fns is None:
                color_print(params, "Initializing inputs...")
                sess.run(input_initializers)


            while True:

                if query_input_fns is None:
                    feed_dict = None
                else:
                    _prompt, _iter_pos, _samp_temp, _end_iter = query_input_fns()
                    feed_dict = {prompt: _prompt,
                                 iter_pos: _iter_pos,
                                 samp_temp: _samp_temp,
                                 end_iter: _end_iter}

                sess.run(enqueue_ops, feed_dict=feed_dict)

                sess.run(computation)
                out = sess.run(outfeed_dequeue_ops)[0]

                for fn in callback_fns:
                    fn(out)
