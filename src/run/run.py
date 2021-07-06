import time
import typing

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2 as summary, variables
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import checkpoint_management

from .dataloader_placement import place_dataloader, infeed_from_session
from .inference import get_infrence_model
from .train import get_train_model
from .utils_run import CheckpointLoaderHook, add_summary, add_histogram, _import_tensor, analyze_model
from .. import tf_wrapper as tfw
from ..dataclass import ModelParameter
from ..mtf_wrapper import reduce_sum
from ..utils_core import color_print

tf1 = tf.compat.v1
Dataset = tf1.data.Dataset


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
                    initial_pos = reduce_sum(initial_pos, output_shape=[])
                    sampling_temperature = _import_tensor(params, args[3], mtf.Shape([initial_pos_dim]), "temperature")
                    sampling_temperature = reduce_sum(sampling_temperature, output_shape=[])
                    end_iterations = _import_tensor(params, args[4], mtf.Shape([initial_pos_dim]), "end_iterations")
                    end_iterations = reduce_sum(end_iterations, output_shape=[])

            if params.train:
                frame_out, token_out, learning_rate, loss, video_loss, \
                token_loss, accuracy, update_ops, debug_gradients_dict = get_train_model(params)(frame_input,
                                                                                                 cat_mask_src,
                                                                                                 cat_mask_tag,
                                                                                                 token_x_input,
                                                                                                 token_y_input,
                                                                                                 frame_mask_src,
                                                                                                 frame_mask_tag,
                                                                                                 token_mask,
                                                                                                 manual_global_step)
            else:
                token_out, frame_out = get_infrence_model(params)(frame_input,
                                                                  cat_mask_src,
                                                                  cat_mask_tag,
                                                                  token_x_input,
                                                                  token_y_input,
                                                                  frame_mask_src,
                                                                  frame_mask_tag,
                                                                  token_mask,
                                                                  initial_pos,
                                                                  sampling_temperature,
                                                                  end_iterations)

            analyze_model(params, time_to_build=(time.time() - start_time), graph=graph)
            color_print(params, "Lowering graph to TensorFlow...")
            start_time = time.time()
            lowering = mtf.Lowering(graph, {params.mesh: params.mesh_impl})
            color_print(params, f"Lowered in {time.time() - start_time:.1f}s")

            if params.train:
                log_dict = {'learning_rate': tfw.cast(learning_rate, tf.float32)}
                if params.use_video:
                    log_dict['video_loss'] = tfw.cast(lowering.export_to_tf_tensor(video_loss), tf.float32)
                if params.use_language:
                    log_dict['token_loss'] = tfw.cast(lowering.export_to_tf_tensor(token_loss), tf.float32)
                if accuracy is not None:
                    log_dict['accuracy'] = tfw.cast(lowering.export_to_tf_tensor(accuracy), tf.float32)

                comput_ops = [lowering.lowered_operation(op) for op in update_ops]

                with tf.control_dependencies(comput_ops):
                    global_step = tf1.train.get_or_create_global_step()

                    step = tfw.mod(tfw.add(manual_global_step, 1),
                                   tfw.constant(params.grad_accumulation, dtype=tf.int64))
                    step = tfw.equal(step, tfw.constant(0, dtype=tf.int64))
                    step = tfw.cast(step, tf.int64)

                    tf_loss = tfw.cast(lowering.export_to_tf_tensor(loss), tf.float32)

                    if params.macro_batching > 1 and params.train:
                        if params.macro_batch_loss_smoothing:
                            tf_loss = tfw.divide(tfw.add(tfw.cast(tf_loss, tf.float32),
                                                         tfw.multiply(tfw.cast(prev_loss, tf.float32),
                                                                      tfw.cast(loop_idx, tf.float32))),
                                                 tfw.cast(tfw.add(loop_idx, 1), tf.float32))
                        params.log_dict_keys = list(log_dict.keys())
                    else:
                        comput_ops.append(add_summary(tf_loss=tf_loss, value=log_dict, global_step=global_step))

                    if params.debug_gradients:
                        for grad_key in debug_gradients_dict.keys():
                            debug_gradients_dict[grad_key] = \
                                tfw.cast(lowering.export_to_tf_tensor(debug_gradients_dict[grad_key]), tf.float32)

                        comput_ops.append(add_histogram(tf_loss=tf_loss, value=debug_gradients_dict,
                                                        global_step=global_step))

                    comput_ops.extend([tfw.assign_add(global_step, step),
                                       tfw.assign_add(manual_global_step, tfw.constant(1, dtype=tf.int64))])

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
                    with tfw.control_dependencies(comput_ops):
                        return ([tfw.add(loop_idx, 1), tf_loss] + [log_dict[key] for key in params.log_dict_keys] +
                                inp_args)
                else:
                    return tfw.group(comput_ops)

            else:  # train == 'sample'
                predictions = {}

                if params.use_video:
                    predictions['frame_out'] = lowering.export_to_tf_tensor(frame_out)
                    predictions['frame_tgt'] = args[0]

                if params.use_language:
                    predictions['token_out'] = lowering.export_to_tf_tensor(token_out)
                    predictions['token_tgt'] = args[1 + int(params.use_video) * 5]

                for key in params.debug_outfeed:
                    predictions[key] = lowering.export_to_tf_tensor(params.debug_outfeed[key])

                predictions = [val if val.dtype == tf.float32 else tf.cast(val, tf.float32) for val in
                               predictions.values()]
                output_shapes.extend([pred.shape for pred in predictions])
                hooks.append(mtf.MtfRestoreHook(lowering))
                return tpu_ops.outfeed_enqueue_tuple(predictions)

        if params.train and params.macro_batching > 1:
            log_len = int(params.use_language) + int(params.use_video) + \
                      int(params.calc_accuracy) * int(params.use_language) + 1
            loop_inputs = [tfw.constant(0, dtype=tf.int32), tfw.constant(0, dtype=tf.float32)]
            loop_inputs = loop_inputs + [tfw.constant(0, dtype=tf.float32) for _ in range(log_len)] + list(args)

            def con(i, *args):
                return tfw.less(i, tfw.constant(params.macro_batching, dtype=tf.int32))

            loop_out = tf.while_loop(cond=con, body=_base_model_fn,
                                     loop_vars=loop_inputs, back_prop=False, parallel_iterations=1)

            tf_loss = loop_out[1]
            log_list = loop_out[2:][:log_len]
            log_dict = {key: val for (key, val) in zip(params.log_dict_keys, log_list)}
            global_step = tf1.train.get_or_create_global_step()

            with tfw.control_dependencies(log_list):
                ret = add_summary(tf_loss=tf_loss, value=log_dict, global_step=global_step)
        else:
            ret = _base_model_fn(*args)

        return ret

    if query_input_fns is None:
        input_initializers, enqueue_ops, infeed_queue = place_dataloader(params, input_fn)
    else:
        enqueue_ops, infeed_queue, (prompt, iter_pos, samp_temp, end_iter) = infeed_from_session(params)

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
            tf.compat.v1.get_default_graph().finalize()
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
                                 end_iter: _end_iter
                                 }

                sess.run(enqueue_ops, feed_dict=feed_dict)

                sess.run(computation)
                out = sess.run(outfeed_dequeue_ops)[0]

                for fn in callback_fns:
                    fn(out)
