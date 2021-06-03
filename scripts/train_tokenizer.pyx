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
cdef int PROCESSES = 16
cdef int VOCAB_SIZE = 65536
cdef int PREFETCH = 128
cdef int CACHE_CAPACITY = 1 << 30
cdef unicode BASE_PATH = "pile2/"
cdef unicode DOWNLOAD_CACHE_PATH = f"{BASE_PATH}download"
cdef unicode BASE_URL = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
# https://the-eye.eu/public/AI/pile/train/%s.jsonl.zst
cdef unicode SPLIT_CHARS = f'{string.digits} \t\n\r\x0b\x0c'
for c in string.punctuation:
    SPLIT_CHARS = SPLIT_CHARS + "\\" + c
cdef unicode SPLIT_REGEX = f"""[{SPLIT_CHARS}]|[^{SPLIT_CHARS}]+"""
cdef list ALWAYS_INCLUDED_TOKENS = [chr(i) for i in range(256)]
cdef unicode OUTPUT_FILE = "tokenizer.json"
cdef int PRINT_INTERVAL = 100 * 1000

# constants
cdef int SPLITS = 30
cdef unicode START = "Starting"
cdef unicode DOWNLOADING = "Downloading"
cdef unicode FINISHED_DOWNLOAD = "Finished downloading"
cdef unicode FILE_EXISTS = "File exists, not downloading"
cdef unicode TEMP_TOKENIZER_PATH = ".tmp.json"


cdef void log(unicode text, unicode log_path, const int pid, const int i):
    with open(log_path, 'a') as f:
        f.write(f'Proc: {pid} | Slice: {i} | Time: {datetime.datetime.now()} | {text}\n')

cdef void file_generator(queue: Queue, lock: threading.Semaphore, const int pid):
    cdef unicode log_path = f"{BASE_PATH}log/{pid}.txt"
    cdef unicode completion = f'{BASE_PATH}/done/{pid}.txt'
    cdef unicode tmp_name = ""
    cdef unicode out = ""
    cdef bytes byte_line = b""
    cdef int splits = 30
    cdef int total = 0
    cdef int idx = 0
    cdef int i = 0
    parse = simdjson.Parser().parse

    with open(log_path, 'w') as f:
        f.write('')

    for i in range(pid, splits, PROCESSES):
        total = 0
        log(START, log_path, pid, i)
        tmp_name = f"{DOWNLOAD_CACHE_PATH}/{i}.zstd"

        if not os.path.exists(tmp_name):
            lock.acquire()
            log(DOWNLOADING, log_path, pid, i)
            os.system(f"wget {BASE_URL.replace('%s', str(i).zfill(2))} -O {tmp_name} -t inf --timeout 15 "
                      f"&& echo 1 > {completion}")
            while not os.path.exists(completion):
                time.sleep(300)
            os.remove(completion)
            log(FINISHED_DOWNLOAD, log_path, pid, i)
            lock.release()
        else:
            log(FILE_EXISTS, log_path, pid, i)

        with open(tmp_name, 'rb') as f:
            for idx, byte_line in enumerate(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f))):
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
                if idx % PRINT_INTERVAL == 0:
                    log(f"{total * 2 ** -20:9.2f}MB", log_path, pid, i)
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
    cdef unicode split_chars = string.punctuation + " \t\n\r\x0b\x0c"
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

    cdef list procs = [multiprocessing.Process(target=file_generator, args=(queue, lock, i)) for i in range(PROCESSES)]
    for p in procs:
        p.start()

    while queue.qsize() < PREFETCH // 2:
        time.sleep(5)

    tokenizer.train_from_iterator(iterator(queue, procs), trainer)

    for p in procs:
        p.join()

    tokenizer.save(TEMP_TOKENIZER_PATH)

    with open(OUTPUT_FILE, 'w', errors='ignore') as w, open(TEMP_TOKENIZER_PATH, 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))

    os.remove(TEMP_TOKENIZER_PATH)
