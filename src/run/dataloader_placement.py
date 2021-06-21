import collections
import json

import jsonpickle
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.framework import ops
from tensorflow.python.tpu import tpu, tpu_feed

from ..dataclass import ModelParameter

tf1 = tf.compat.v1
Dataset = tf1.data.Dataset


def place_dataloader(params: ModelParameter, input_fn):

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

        curran_stats = {'steps': params.current_step, 'ctx': params.n_ctx,
                        'slice_count': len(hosts_to_hold_ds),
                        'interleave_size': params.interleaved_datasets,
                        'batch_size': params.train_batch_size,
                        'grad_accumulation': params.grad_accumulation,
                        'token_patch_size': params.token_patch_size
                        }

        size_dump = jsonpickle.dumps(_run_log + [curran_stats], indent=4)
        with tf.io.gfile.GFile(f"{params.model_path}/model_size.info", 'w') as f:
            f.write(size_dump)

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
            # options.experimental_optimization.hoist_random_uniform = True
            options.experimental_optimization.map_and_batch_fusion = True
            options.experimental_optimization.map_and_filter_fusion = False
            options.experimental_optimization.map_fusion = True
            options.experimental_optimization.map_parallelization = True
            # options.experimental_optimization.map_vectorization.enabled = True
            # options.experimental_optimization.map_vectorization.use_choose_fastest = True
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

    return input_initializers, enqueue_ops, infeed_queue


def infeed_from_session(params: ModelParameter):

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

    place_holders = [prompt, iter_pos, samp_temp, end_iter]
    return enqueue_ops, infeed_queue, place_holders