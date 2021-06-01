#!python
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

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
import jsonlines
import jsonpickle
import simdjson
import zstandard
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

# config
cdef int PROCS = 16
cdef int VOCAB_SIZE = 65536
cdef int PREFETCH = 128
cdef int CACHE_CAPACITY = 1 << 30
cdef unicode BASE_PATH = "pile2/"

# constants
cdef int SPLITS = 30
cdef unicode BASE_URL = 'https://the-eye.eu/public/AI/pile/train/%s.jsonl.zst'
# http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst
cdef unicode START = "Starting"
cdef unicode DOWNLOADING = "Downloading"
cdef unicode FINISHED_DOWNLOAD = "Finished downloading"

cpdef parser(x: str):
    return simdjson.Parser().parse(x.encode()).as_dict()

cpdef write(text: str, list total):
    out = ftfy.fix_text(text).replace('    ', '\t')
    total[0] += len(out)
    return out

cpdef log(text: str, log_path: str, const int pid, const int i):
    with open(log_path, 'a') as f:
        f.write(f'Proc: {pid} | Slice: {i} | Time: {datetime.datetime.now()} | {text}\n')

cpdef file_generator(queue: Queue, lock: threading.Semaphore, int pid, int procs, base_path: str):
    tmp_name = f"{base_path}download/{pid}.zstd"
    log_path = f"{base_path}log/{pid}.txt"
    completion = f'{base_path}/done/{pid}.txt'
    cdef int splits = 30
    cdef list total = [0]
    cdef int idx = 0

    with open(log_path, 'w') as f:
        f.write('')

    for i in range(pid, splits, procs):
        total[0] = 0
        log(START, log_path, pid, i)
        lock.acquire()
        log(DOWNLOADING, log_path, pid, i)
        os.system(f"wget {BASE_URL.replace('%s', str(i).zfill(2))} -O {tmp_name} -t inf --timeout 15 "
                  f"&& echo 1 > {completion}")
        while not os.path.exists(completion):
            time.sleep(300)
        os.remove(completion)
        log(FINISHED_DOWNLOAD, log_path, pid, i)
        lock.release()
        with open(tmp_name, 'rb') as f:
            read = jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)), loads=parser)
            for idx, item in enumerate(read):
                item = item['text']
                if isinstance(item, list):
                    for itm in item:
                        queue.put(write(itm, total))
                else:
                    queue.put(write(item, total))
                if idx % 100000 == 0:
                    log(f"{total[0] / 2 ** 20:9.2f}MB", log_path, pid, i)
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

cpdef main():
    for path in ('', 'download', 'log', 'done'):
        if not os.path.exists(BASE_PATH + path):
            os.mkdir(BASE_PATH + path)

    cdef unicode backslash = "\\"

    cdef unicode chars = f'{string.digits} \t\n\r\x0b\x0c'
    for c in string.punctuation:
        chars = chars + backslash + c

    regex = Regex(f"""[{chars}]|[^{chars}]+""")
    cdef list special_tokens = [chr(i) for i in range(256)]

    # set up tokenizer
    tokenizer = Tokenizer(BPE(unk_token='\x01', cache_capacity=CACHE_CAPACITY, merges=None, dropout=None))
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=VOCAB_SIZE)
    tokenizer.pre_tokenizer = Split(regex, 'isolated')

    # train and save
    manger = multiprocessing.Manager()
    queue = manger.Queue(PREFETCH)
    lock = manger.Semaphore()

    cdef list procs = [multiprocessing.Process(target=file_generator, args=(queue, lock, i, PROCS, BASE_PATH))
                       for i in range(PROCS)]
    for p in procs:
        p.start()

    while queue.qsize() < PREFETCH // 2:
        time.sleep(5)
    tokenizer.train_from_iterator(iterator(queue, procs), trainer)
    for p in procs:
        p.join()

    tokenizer.save(f".tmp.json")

    with open(f"tokenizer.json", 'w', errors='ignore') as w, open(f".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))

if __name__ == "__main__":
    main()
