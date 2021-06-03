#!python
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False
# cython: linetrace=False
# cython: language_level=3

import datetime
import io
import multiprocessing
import os
import string
import threading
import time
import typing
from queue import Queue

import ftfy
import jsonpickle
import simdjson
import zstandard
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

# config
DEF PROCESSES = 16
DEF VOCAB_SIZE = 65536UL
DEF PREFETCH = 128
DEF CACHE_CAPACITY = 1UL << 30
DEF BASE_PATH = "pile2/"
DEF DOWNLOAD_CACHE_PATH = "pile2/download"
DEF BASE_URL = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
DEF PRINT_INTERVAL = 100000
# https://the-eye.eu/public/AI/pile/train/%s.jsonl.zst


cdef void log(unicode text, unicode log_path, const unsigned char pid, const unsigned char i):
    with open(log_path, 'a') as f:
        f.write(f'Proc: {pid} | Slice: {i} | Time: {datetime.datetime.now()} | {text}\n')

cdef void file_generator(queue: Queue, lock: threading.Semaphore, const unsigned char pid):
    cdef unicode log_path = f"{BASE_PATH}log/{pid}.txt"
    cdef unicode completion = f'{BASE_PATH}/done/{pid}.txt'
    cdef unicode tmp_name = ""
    cdef unicode out = ""
    cdef bytes byte_line = b""
    cdef unsigned long long total = 0
    cdef unsigned long idx = 0
    cdef unsigned char i = 0
    stream_reader = zstandard.ZstdDecompressor().stream_reader
    parse = simdjson.Parser().parse

    with open(log_path, 'w') as f:
        f.write('')

    for i in range(pid, 30, PROCESSES):
        total = 0
        log("Starting", log_path, pid, i)
        tmp_name = f"{DOWNLOAD_CACHE_PATH}/{i}.zstd"

        if not os.path.exists(tmp_name):
            lock.acquire()
            log("Downloading", log_path, pid, i)
            os.system(f"wget {BASE_URL.replace('%s', str(i).zfill(2))} -O {tmp_name} -t inf --timeout 15 "
                      f"&& echo 1 > {completion}")
            while not os.path.exists(completion):
                time.sleep(300)
            os.remove(completion)
            log("Finished downloading", log_path, pid, i)
            lock.release()
        else:
            log("File exists, not downloading", log_path, pid, i)

        with open(tmp_name, 'rb') as f:
            for idx, byte_line in enumerate(io.BufferedReader(stream_reader(f))):
                item = parse(byte_line)['text']
                if isinstance(item, list):
                    for out in item:
                        out = ftfy.fix_text(out).replace('    ', '\t')
                        total += len(out)
                        queue.put(out)
                else:
                    out = ftfy.fix_text(item).replace('    ', '\t')
                    total += len(out)
                    queue.put(out)
                if idx % PRINT_INTERVAL  == 0:
                    log(f"{total:15,}B", log_path, pid, i)
        os.remove(tmp_name)

def iterator(queue: Queue, procs: typing.List[multiprocessing.Process]):
    die = False
    while True:
        try:
            yield queue.get(timeout=60)
        except:
            die = True
            for p in procs:
                if p.is_alive():
                    die = False
                    break
            if die:
                break

cpdef void main():
    for path in ('', 'download', 'log', 'done'):
        if not os.path.exists(BASE_PATH + path):
            os.mkdir(BASE_PATH + path)

    # set up tokenizer
    cdef unicode split_chars = string.digits + " \t\n\r\x0b\x0c"
    for c in string.punctuation:
        split_chars += '\\' + c
    regex = Regex(f"""[{split_chars}]|[^{split_chars}]+""")
    tokenizer = Tokenizer(BPE(unk_token='\x01', cache_capacity=CACHE_CAPACITY, merges=None, dropout=None))
    trainer = BpeTrainer(special_tokens=[chr(i) for i in range(256)], vocab_size=VOCAB_SIZE)
    tokenizer.pre_tokenizer = Split(regex, 'isolated')

    # train and save
    manger = multiprocessing.Manager()
    queue = manger.Queue(PREFETCH)
    lock = manger.Semaphore()

    cdef tuple procs = tuple([multiprocessing.Process(target=file_generator, args=(queue, lock, i)) for i in range(PROCESSES)])
    for p in procs:
        p.start()

    while queue.qsize() < PREFETCH // 2:
        time.sleep(5)

    tokenizer.train_from_iterator(iterator(queue, procs), trainer)

    for p in procs:
        p.join()

    tokenizer.save(".tmp.json")

    with open("tokenizer.json", 'w', errors='ignore') as w, open(".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))

    os.remove(".tmp.json")
