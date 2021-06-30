"""
tokenization to bpe or character embeddings of text datasets
make sure to first run train_tokenizer to get the preprocessed files locally
"""

import argparse
import io
import os
import shutil
import time
import random
import multiprocessing

import jsonlines
import requests
import simdjson
import tensorflow as tf
import zstandard
from google.cloud import storage
from transformers import GPT2TokenizerFast


DEF NAME = "gpt2-bpe"
DEF INT64 = 1
DEF BUCKET_NAME = "obst-euw4a-aa"
DEF OUTPUT_DIR = "the-fixed-gpt2-bpe-pile/"
DEF PROCS = 12
DEF SERVICE_ACCOUNT_JSON_PATH = "a.json"
DEF BUFFER_SIZE = 2 ** 24
DEF PRINTERVALL = 16

cdef void create_tfrecords(unsigned short pid):
    cdef unicode prefix = f"{'int64' if INT64 else 'bytes'}_{NAME}_"

    bucket = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON_PATH).get_bucket(BUCKET_NAME)
    encode = (GPT2TokenizerFast.from_pretrained('gpt2') if INT64 else str).encode

    cdef unsigned short splits = 30
    cdef unsigned short i = 0
    cdef unicode txt = ""
    cdef unicode filename = ""
    cdef unsigned long long processed_chars = 0
    cdef unsigned long tfrecord_count = 0

    cdef unsigned long last_write = time.time()
    cdef unsigned long start_time = time.time()

    for i in range(pid, splits, PROCS):
        with open(f'{i}.txt', 'r', BUFFER_SIZE * 2) as f:
            while True:
                txt = f.read(BUFFER_SIZE)
                if not txt:
                    break
                processed_chars += BUFFER_SIZE
                joined = encode(txt)

                filename = f"{prefix}{tfrecord_count:_>6d}_{processed_chars}_{len(joined)}.tfrecord"

                with tf.io.TFRecordWriter(filename) as writer:
                    if INT64:
                        feature = {"text": tf.train.Feature(int64_list=tf.train.Int64List(value=joined))}
                    else:
                        feature = {"text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[joined]))}
                    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(tf_example.SerializeToString())

                bucket.blob(f'{OUTPUT_DIR}{filename}').upload_from_filename(filename)
                tfrecord_count += 1
                if tfrecord_count % PRINTERVALL == 0:
                    print(f"[{pid:{len(str(PROCS))}d}/{PROCS}] Processed: {processed_chars} - Total: {time.time()-start_time:.0f}s - Since last write: {time.time()-last_write:.0f}s")
                last_write = time.time()

cpdef main():
    processes = [multiprocessing.Process(target=create_tfrecords, args=(pid,)) for pid in range(PROCS)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
