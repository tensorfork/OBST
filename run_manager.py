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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('run_command', type=str)
    parser.add_argument('tpu_name', type=str)
    parser.add_argument('tpu_type', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('preemptible', type=str)

    args = parser.parse_args()

    run_command = args.run_command
    tpu_name = args.tpu_name
    tpu_type = args.tpu_type
    model_path = args.model_path
    preemptible = str2bool(args.preemptible)

    tpu_client = TPUServiceAPI(project='mlops-engine')

    out_io = tf.io.gfile.GFile(f"{model_path}/run_config.log", 'w')

    try:
        tpu_log = tpu_client.create(tpu_name, mesh=tpu_type, tf_version='1.15.5', zone='europe-west4-a',
                                    cidrblock=None, preemptible=preemptible, wait=True)

        out_io.write(f"{tpu_log}\n\n\n")

        ready = False
        while not ready:
            time.sleep(15)
            ready = tpu_client.is_tpu_ready(tpu_name)

        pro = subprocess.Popen(run_command, stdout=out_io, shell=True, preexec_fn=os.setsid)

        done = False

        while not done:
            time.sleep(600 + random.randint(0, 300))

            health = tpu_client.is_tpu_ready(tpu_name)
            if pro.poll() is not None:
                if health['healthy']:
                    done = True

            if not done and not health['healthy']:
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

                tpu_log = tpu_client.recreate(tpu_name, mesh=tpu_type, tf_version='1.15.5', zone='europe-west4-a',
                                              cidrblock=None, preemptible=preemptible, wait=True)

                out_io.write(f"\n\n\n{tpu_log}\n\n\n")

                ready = False
                while not ready:
                    time.sleep(15)
                    ready = tpu_client.is_tpu_ready(tpu_name)

                pro = subprocess.Popen(run_command, stdout=out_io, shell=True, preexec_fn=os.setsid)
    except:
        out_io.write(f"\n\n\nrun_manager has crashed\n\n\n")

    try:
        tpu_log = tpu_client.delete(tpu_name)
        out_io.write(f"\n\n\n{tpu_log}")
    except:
        out_io.write(f"\n\n\nFailed to Delete the TPU")

    out_io.flush()
    out_io.close()
