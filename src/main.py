"""
"Sub Main" that contains one function to start the training loop.
"""

import argparse
import re
import time

import jsonpickle
import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf2
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.tpu.device_assignment import device_assignment
from tensorflow.python.tpu.topology import Topology
from tensorflow_estimator.python.estimator import estimator as estimator_lib

from .dataclass import ModelParameter
from .inputs import dataset, gpt_neo_input
from .interface import gen_sample_fn, get_command_line_input_and_output_fn, get_similarity_input_and_output_fn
from .rest_api import get_api_input_and_output_fn
from .run.run import computation_func

tf = tf2.compat.v1
tpu = tf.tpu


def sample_output_fn(params: ModelParameter):
    return None, gen_sample_fn(params)


def raise_str(arg: str):
    raise ValueError(arg)


RUN_MODE_FNS = {'debug_old': sample_output_fn,
                'sample': sample_output_fn,
                'web_api': get_api_input_and_output_fn,
                'debug': get_similarity_input_and_output_fn,
                'query': get_command_line_input_and_output_fn,
                'train': lambda x: raise_str("Train should've been caught by code above. Something is wrong.")}


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
        _params = f.read()
    _params = jsonpickle.loads(_params)
    params = ModelParameter(_params)
    params.web_workers = args.workers
    params.train = args.run_mode == 'train'
    params.debug_sample = args.run_mode == 'debug_old'
    params.debug_gradients = args.debug_grad is not None

    # Read params of model
    if params.train:
        param_dump = jsonpickle.dumps(_params, indent=4)
        with tf.io.gfile.GFile(f"{params.model_path}/run_config_{int(time.time())}.json", 'w') as f:
            f.write(param_dump)

    params.current_step = int(estimator_lib._load_global_step_from_checkpoint_dir(params.model_path))

    # If run mode == sample, set the batch size to one
    if not params.train:
        if params.debug_sample:
            params.train_batch_size = 2
            params.use_autoregressive_sampling = True
            params.sampling_temperature = 0
        else:
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
    # Expand attention types param

    mtf_mesh_shape = mtf.convert_to_shape(params.mesh_shape)
    params.layout_rules = mtf.convert_to_layout_rules(params.layout)

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.tpu)
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    tpu_cluster_spec = tpu_cluster_resolver.cluster_spec()

    if tpu_cluster_spec:
        session_config.cluster_def.CopyFrom(tpu_cluster_spec.as_cluster_def())

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
            params.mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(mtf_mesh_shape, params.layout_rules,
                                                               None, params.d_assignment)

        if params.train:
            summary_writer = summary.create_file_writer(params.model_path)
            with summary_writer.as_default(), (summary.always_record_summaries()):
                computation_func(params,
                                 input_fn,
                                 session_config,
                                 tpu_cluster_resolver,
                                 [lambda x: print(f"Current step: {x}")] * params.debug_train_step)
            return

        input_fns, output_fn = RUN_MODE_FNS[args.run_mode](params)

        computation_func(params,
                         input_fn,
                         session_config,
                         tpu_cluster_resolver,
                         [output_fn],
                         input_fns)
