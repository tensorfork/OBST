"""
"Sub Main" that contains one function to start the training loop.
"""

import argparse
import json
import re

import mesh_tensorflow as mtf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import tpu
from tensorflow.python.tpu import tpu_config, tpu_estimator
from tensorflow.python.tpu.device_assignment import device_assignment
from tensorflow.python.tpu.topology import Topology
from tensorflow_estimator.python.estimator import estimator as estimator_lib

from .dataclass import ModelParameter
from .inputs import dataset, gpt_neo_input
from .train import model_fn


def main(args: argparse.Namespace) -> None:
    """
    Given previously captured arguments, this function runs the following steps (in order):
    * Load given session_config
    * initialize data loader
    * create model graph
    * start training loop.
    :param args: argparse arguments from the parent main function
    :return: None
    """
    # Setup logging
    model_path = args.model if args.model.endswith(".json") else f"session_configs/{args.model}.json"
    with open(model_path) as f:
        _params = json.load(f)
    params = ModelParameter(_params)
    params.train = args.run_mode == 'train'
    # Read params of model

    json.dump(_params, tf.io.gfile.GFile(f"{params.model_path}/run_config.json", 'w'))

    params.current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))

    # If run mode == sample, set the batch size to one
    if not params.train:
        params.train_batch_size = 1
        params = ModelParameter(params)

    # Fetch appropriate input functions
    if params.model_mode == 'jannet':
        input_fn = dataset
    elif params.model_mode == 'gpt':
        input_fn = gpt_neo_input

        # Set params for text only GPT mode.
        params.use_language = True
        params.use_video = False
        params = ModelParameter(params)

    else:
        raise ValueError(f"model_mode need to be 'jannet' or 'gpt' {params.model_mode}, "
                         "is a not supported option.")

    # Add to params: auto_layout, auto_layout_and_mesh_shape, use_tpu, num_cores
    mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    params.num_cores = mesh_shape.size
    params.layout_rules = mtf.convert_to_layout_rules(params.layout)
    params.use_tpu = True if not args.tpu is None else False
    params.gpu_ids = args.gpu_ids
    # Expand attention types param
    params.predict = args.predict

    if args.dry:
        inp = {'token_x': tf.zeros([1]), 'token_y': tf.zeros([1]), 'frame': tf.zeros([1]), 'vid_msk': tf.zeros([1]),
               'tkn_msk': tf.zeros([1])
               }
        model_fn(inp, "TRAIN", params.dict())
        return

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

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
    options.experimental_distribute.auto_shard = True

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    with tf.Graph().as_default():

        with tf.Session(target=tpu_cluster_resolver.master(), config=session_config) as sess:
            tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

            all_devices = sess.list_devices()

            cpus = []
            for d in all_devices:
                if d.device_type == 'CPU':
                    cpus += [re.sub('device:CPU', 'cpu', d.name)]

            cpu_devices = []
            for c in cpus:
                m = re.match('/job:(.*)/replica:(.*)/task:(.*)/.*', c)
                cpu_devices.append((m.group(1), int(m.group(2)), int(m.group(3)), c))

            cpu_devices = [_[3] for _ in sorted(cpu_devices)]
            params.cpu_devices = [n for n in cpu_devices if 'coordinator' not in n]

            topology = sess.run(tpu.initialize_system())
            topo_object = Topology(serialized=topology)

            params.num_cores = int(np.prod(topo_object.mesh_shape))
            params.num_hosts = int(topo_object.num_tasks)
            params.num_cores_per_host = int(params.num_cores // params.num_hosts)
            if params.num_cores_per_host != int(topo_object.num_tpus_per_task):
                raise ValueError

            params.d_assignment = device_assignment(topology, num_replicas=params.num_cores,
                                                    computation_shape=[1, ] * mtf.utils.topology_rank(topology))
            params.mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(mesh_shape, params.layout_rules,
                                                               None, params.d_assignment)

    def get_dataset(params):
        params = ModelParameter(params)
        return input_fn(params, params.train_batch_size, 0, 1).prefetch(params.buffer_size).with_options(options)

    config = tpu_config.TPUConfig(num_shards=mesh_shape.size,
                                  iterations_per_loop=params.iterations,
                                  num_cores_per_replica=1,
                                  per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST)
    config = tpu_config.RunConfig(cluster=tpu_cluster_resolver,
                                  model_dir=params.model_path,
                                  save_checkpoints_steps=None,  # Disable the default saver
                                  save_checkpoints_secs=None,  # Disable the default saver
                                  log_step_count_steps=params.iterations,
                                  save_summary_steps=params.iterations,
                                  tpu_config=config)
    estimator = tpu_estimator.TPUEstimator(use_tpu=params.use_tpu,
                                           model_fn=model_fn,
                                           config=config,
                                           train_batch_size=params.train_batch_size,
                                           predict_batch_size=1,
                                           params=params.dict())

    current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))

    if current_step < params.train_steps:
        estimator.train(input_fn=get_dataset, max_steps=10 ** 9)
