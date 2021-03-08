import subprocess
import argparse
import hashlib
import json
import math
import time
import os

import numpy as np

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('base_config', type=str,
                        help="The path to the .json config with will be use as the bases for this run.")
    parser.add_argument('tpu_start_id', type=int, help="The tpu ID at with the TPU IDs will start.")
    parser.add_argument('--run_config', default='', type=str, help="The path to the .json config that continents "
                                                                   "the Hyperparameters to be run. The config must contain a"
                                                                   " dict in with each entry is a list, the list continents "
                                                                   "the different Hyperparameters for variable")
    parser.add_argument('--run_name_prefix', type=str, default='gs://text-datasets/video-transformer/')
    parser.add_argument('--number_of_repetitions', type=int, default=1, help="The number of times the same "
                                                                             "parameters will get tested.")
    parser.add_argument('--repetition_start_idx', type=int, default=0)
    parser.add_argument('--use_preemptible', type=str, default='true')
    parser.add_argument('--tpu_type', type=str, default='v3-8')
    parser.add_argument('--zone', type=str, default='europe-west4-a')
    parser.add_argument('--network', type=str, default='tpu-euw4a')
    parser.add_argument('--start_up_sleep', type=int, default=0)
    parser.add_argument('--project', type=str, default='mlops-engine')
    parser.add_argument('--use_manager', type=str, default='False')

    args = parser.parse_args()

    tpu_type = args.tpu_type
    tpu_type_str = '"' + tpu_type + '"'

    with open(args.base_config) as f:
        base_config = json.load(f)

    if args.run_config != "":
        with open(args.run_config) as f:
            run_config = json.load(f)
    else:
        run_config = {}

    if not os.path.exists("buffer_configs/"):
        os.makedirs("buffer_configs/")

    tpu_id = args.tpu_start_id
    run_config_key = list(run_config.keys())

    _key = [np.arange(len(run_config[key])) for key in run_config_key]
    key_pos = np.meshgrid(*_key, sparse=False)
    key_pos = np.stack(key_pos, axis=-1)
    _shape = key_pos.shape
    key_pos = np.reshape(key_pos, newshape=(np.prod(_shape[:-1]), _shape[-1]))

    for pos in key_pos:

        copy_base_config = base_config.copy()

        for idx, key in enumerate(run_config_key):
            copy_base_config[key] = run_config[key][pos[idx]]

        for repetition_idx in range(args.repetition_start_idx, args.number_of_repetitions):
            tpu_name = f"tpu-{tpu_type}-{args.network}-{tpu_id}"

            cors = int(str(tpu_type).split('-')[-1])
            if cors == 8:
                tpu_range = f"10.48.{tpu_id}.0/29"
            else:
                cidr = int(32 + 2 - math.log2(cors))
                _tpu_id = tpu_id + 2

                tpu_range = f"10.{_tpu_id}.0.0/{cidr}"

            run_name = f"-run={repetition_idx}"
            run_name = "-".join([f"{key}={copy_base_config[key]}" for key in run_config_key]) + run_name
            run_name = run_name.replace(' ', '_').replace("'", '').replace(":", '=').replace(",", '-')
            run_name = run_name.replace('[', '|').replace(']', '|')

            copy_base_config['model_path'] = args.run_name_prefix + run_name

            with open(f"buffer_configs/{tpu_id}_{run_name}.json", 'w+') as w:
                w.write(json.dumps(copy_base_config))

            experiment_command = f"python3 ../main.py --model buffer_configs/{tpu_id}_{run_name}.json --tpu {tpu_name}"
            delete_command = f"pu delete {tpu_name} --yes"
            tpu_creat_command = f"gcloud compute tpus create {tpu_name} --zone {args.zone} " \
                                f"--range {tpu_range} --network {args.network} --version 1.15.5 " \
                                f"--accelerator-type {tpu_type_str} --project {args.project}"

            if str2bool(args.use_preemptible):
                tpu_creat_command = tpu_creat_command + " --preemptible"

            if str2bool(args.use_manager):
                comm = f"python3 run_manager.py '{experiment_command}' {tpu_name} {tpu_type} {args.zone} " \
                       f"{args.network} {args.run_name_prefix + run_name} {str2bool(args.use_preemptible)}"
            else:
                comm = f"({tpu_creat_command} && {experiment_command}) ; {delete_command}"

            if len(run_name) > 66:
                run_name = hashlib.sha256(run_name.encode('utf-8')).hexdigest()

            prosses_name = f"tpu_id:{tpu_id}--{run_name}"


            subprocess.run(['screen', '-dmS', prosses_name, 'bash', '-c', comm])

            tpu_id = tpu_id + 1

            print(f"Creating {prosses_name}")
            time.sleep(args.start_up_sleep)
