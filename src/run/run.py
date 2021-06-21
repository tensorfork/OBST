import collections
import json
import time
import typing

import jsonpickle
import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2 as summary, variables
from tensorflow.python.tpu import tpu, tpu_feed
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import checkpoint_management

from src.dataclass import ModelParameter
from src.model import build
from src.mtf_wrapper import constant_scalar, log
from src.optimizers import get_optimizer
from src.utils_core import color_print
from src.utils_mtf import concat, pad, slice, to_fp32, weighted_add

from src.run.dataloader_placement import place_dataloader
from src.run.inference import autoregressive_model
from src.run.utils_run import CheckpointLoaderHook, add_summary, add_histogram, _import_tensor

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
                token_out, frame_out = autoregressive_model(params,
                                                            frame_input,
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
                            tf_loss = tf.cast(tf_loss, tf.float32)
                            tf_loss += tf.cast(prev_loss, tf.float32) * tf.cast(loop_idx, tf.float32)
                            tf_loss /= tf.cast(1 + loop_idx, tf.float32)
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

    if query_input_fns is None:
        input_initializers, enqueue_ops = place_dataloader(params, input_fn)

    else:
        num_cores = params.mesh_impl.device_assignment.num_replicas
        d_assignment = params.mesh_impl.device_assignment
        ordered_ordinals = []
        ordered_hosts = []

        for pnum in range(num_cores):
            physical_pnum = params.mesh_impl.l2p(pnum)
            host_device = d_assignment.host_device(replica=physical_pnum)
            ordered_hosts.append(host_device)

            # For MTF, there's always 1 core per replica. So logical_core=0.
            ordered_ordinals.append(d_assignment.tpu_ordinal(replica=physical_pnum, logical_core=0))

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
