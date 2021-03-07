import subprocess
import argparse
import hashlib
import json
import math
import random
import time
import signal
import os

import tensorflow as tf
from tpuapi import TPUServiceAPI


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class GFile:

    def __init__(self, name, mode):
        self.file = tf.io.gfile.GFile(name, mode)

    def fileno(self):
        return 9

    def write(self, data):
        self.file.write(data)
        return 3

    def close(self):
        self.file.flush()
        self.file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('run_command', type=str)
    parser.add_argument('tpu_name', type=str)
    parser.add_argument('tpu_type', type=str)
    parser.add_argument('zone', type=str)
    parser.add_argument('network', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('preemptible', type=str)

    args = parser.parse_args()

    run_command = args.run_command
    tpu_name = args.tpu_name
    tpu_type = args.tpu_type
    model_path = args.model_path
    preemptible = str2bool(args.preemptible)

    cors = int(str(tpu_type).split('-')[-1])
    tpu_id = int(str(tpu_name).split('-')[-1])

    if cors == 8:
        tpu_range = f"10.48.{tpu_id}.0/29"
    else:
        cidr = int(32 + 2 - math.log2(cors))
        _tpu_id = tpu_id + 2

        tpu_range = f"10.{_tpu_id}.0.0/{cidr}"

    tpu_client = TPUServiceAPI(project='mlops-engine')

    out_io = GFile(f"{model_path}/run.log", 'w')


    def wait_for_tpu():
        ready = False
        ready_count = 0

        while not ready:
            time.sleep(15)
            ready = tpu_client.is_tpu_ready(tpu_name)['healthy']
            ready_count = ready_count + 1

            if ready_count > 15:
                ready_count = 0
                tpu_log = tpu_client.recreate(tpu_name, mesh=tpu_type, tf_version='1.15.5',
                                              zone=args.zone, cidrblock=tpu_range,
                                              preemptible=preemptible, wait=True, network=args.network)

                out_io.write(f"\n\n\n{tpu_log}\n\n\n")

    try:
        tpu_log = tpu_client.create(tpu_name, mesh=tpu_type, tf_version='1.15.5', zone=args.zone,
                                    cidrblock=tpu_range, preemptible=preemptible, wait=True, network=args.network)

        out_io.write(f"{tpu_log}\n\n\n")

        wait_for_tpu()

        pro = subprocess.Popen(run_command, stdout=out_io, stderr=out_io, shell=True, preexec_fn=os.setsid)

        done = False

        while not done:
            time.sleep(300 + random.randint(0, 300))

            health = tpu_client.is_tpu_ready(tpu_name)
            if pro.poll() is not None:
                if health['healthy']:
                    done = True

            if not done and not health['healthy']:
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

                tpu_log = tpu_client.recreate(tpu_name, mesh=tpu_type, tf_version='1.15.5',
                                              zone=args.zone, cidrblock=tpu_range,
                                              preemptible=preemptible, wait=True, network=args.network)

                out_io.write(f"\n\n\n{tpu_log}\n\n\n")

                wait_for_tpu()

                pro = subprocess.Popen(run_command, stdout=out_io, stderr=out_io, shell=True, preexec_fn=os.setsid)
    except Exception as e:
        out_io.write(f"\n\n\nrun_manager has crashed\n{e}\n\n\n")

    try:
        tpu_log = tpu_client.delete(tpu_name)
        out_io.write(f"\n\n\n{tpu_log}")
    except:
        out_io.write(f"\n\n\nFailed to Delete the TPU")

    out_io.close()
