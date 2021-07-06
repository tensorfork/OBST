"""
"Main" script that parses arguments and starts functions that actually build the model graph and start
training if so desired.
"""

import argparse

import tensorflow as tf

from src.main import main

if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on")
    parser.add_argument("--model", type=str, default=None, help="JSON file that contains model parameters.")
    parser.add_argument("--worker", type=int, default=1, help="Number of workers in WebAPI.")
    parser.add_argument("--run_mode", type=str, default="train", help="'train', 'sample', 'query', 'debug' or"
                                                                      " 'debug_old'")
    parser.add_argument("--debug_grad", help="Log the gradients to tensorbord.")

    args = parser.parse_args()

    if args.run_mode not in ['train', 'sample', 'query', 'debug', 'debug_old']:
        raise ValueError(f"'{args.run_mode}' is not a supported argument for"
                         f" --run_mode, please use 'train', 'sample' , 'query' 'debug' or 'debug_old'.")

    main(args)
