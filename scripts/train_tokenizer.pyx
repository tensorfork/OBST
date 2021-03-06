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

import jsonpickle
from ftfy import fix_text
from simdjson import Parser
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from zstandard import ZstdDecompressor

# config
DEF PROCESSES = 8
DEF VOCAB_SIZE = 65536UL
DEF PREFETCH = 128
DEF CACHE_CAPACITY = 1UL << 20
DEF BASE_PATH = "/mnt/wolf/"
DEF DOWNLOAD_CACHE_PATH = "/mnt/wolf/"
DEF BASE_URL = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
# https://the-eye.eu/public/AI/pile/train/%s.jsonl.zst
DEF PRINT_INTERVAL = 100000
DEF SPLITS = 30
DEF REMOVE_INTERMEDIATE = False
DEF REMOVE_LAST_INTERMEDIATE = False
DEF STREAM = False  # if less than 2TB memory are available

cdef void log(unicode text, const unsigned char pid, const unsigned char i):
    with open(f"{BASE_PATH}log/{pid}.txt", 'a') as f:
        f.write(f'Proc: {pid} | Slice: {i} | Time: {datetime.datetime.now()} | {text}\n')

cdef unicode download_command(const unsigned char i, unicode tmp_name):
    url = "http://eaidata.bmk.sh/data/" if i % 2 else "https://the-eye.eu/public/AI/"
    return f"wget {url}/pile/train/{i:02d}.jsonl.zst -O {tmp_name}.tmp -t inf --timeout 15 && mv {tmp_name}.tmp {tmp_name}"

cdef void sleep_till_exists(unicode file_path):
    while not os.path.exists(file_path):
        time.sleep(300)

cdef void wait_for_bash(const unsigned char i, const unsigned char pid, unicode start, unicode end, unicode command):
    cdef unicode completion = f'{BASE_PATH}done/{pid}.txt'
    log(start, pid, i)
    os.system(f'{command} && echo 1 > {completion}')
    sleep_till_exists(completion)
    os.remove(completion)
    log(end, pid, i)

cdef void locked_execution(const unsigned char i, const unsigned char pid, lock: threading.Semaphore, unicode start,
                           unicode end, unicode command):
    lock.acquire()
    wait_for_bash(i, pid, start, end, command)
    lock.release()

cdef unsigned char check_files(list paths):
    cdef unicode path = ""
    for path in paths:
        if os.path.exists(path):
            return 1
    return 0

cdef void extract(const unsigned char pid, lock: threading.Semaphore):
    cdef unicode tmp_name = f"{DOWNLOAD_CACHE_PATH}{pid}"
    cdef unicode tmp_zstd = tmp_name + '.zst'
    print(f"extract {pid} sleep")
    if check_files([tmp_name, tmp_name + '.txt']):
        print(f"extract {pid} no")
        return
    sleep_till_exists(tmp_zstd)
    print(f"extract {pid} start")
    locked_execution(pid, pid, lock, "Extracting", "Finished extraction", f"unzstd {tmp_zstd} && mv {tmp_name} {tmp_name}.jsonl")
    if REMOVE_INTERMEDIATE:
        os.remove(tmp_zstd)

cdef void download(const unsigned char i, const unsigned char pid, lock: threading.Semaphore):
    cdef unicode tmp_name = f"{DOWNLOAD_CACHE_PATH}{pid}"
    cdef unicode tmp_zstd = tmp_name + '.zst'
    if check_files([tmp_zstd] + [tmp_name, tmp_name + '.txt'] * (not STREAM)):
        return
    locked_execution(i, pid, lock, "Downloading", "Finished download", download_command(i, tmp_zstd))

cdef unicode fix_string(bytes byte_line, const unsigned short pid, const unsigned short i, const unsigned long idx,
                        unsigned long long * total):
    cdef unicode out = Parser().parse(byte_line)['text']
    total[0] += len(out)
    if idx % PRINT_INTERVAL == 0:
        log(f"{total[0]:15,}B", pid, i)
    out = fix_text(out)
    out = out.replace('    ', '\t')
    return out

cdef void file_generator(queue: Queue, lock: threading.Semaphore, const unsigned char pid):
    cdef unicode log_path = f"{BASE_PATH}log/{pid}.txt"
    cdef unicode tmp_name = ""
    cdef bytes byte_line = b""
    cdef unsigned long long total = 0
    cdef unsigned long idx = 0
    cdef unsigned char i = 0
    stream_reader = ZstdDecompressor().stream_reader

    with open(log_path, 'w') as f:
        f.write('')

    for i in range(pid, SPLITS, PROCESSES):
        total = 0
        tmp_name = f"{DOWNLOAD_CACHE_PATH}{i}.zst"
        log("Starting", pid, i)
        download(i, pid, lock)

        with open(tmp_name, 'rb') as f:
            for idx, byte_line in enumerate(io.BufferedReader(stream_reader(f))):
                queue.put(fix_string(byte_line, pid, i, idx, &total))
        if REMOVE_LAST_INTERMEDIATE:
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

cdef jsonl_to_txt(const unsigned short i, lock: threading.Lock):
    cdef unicode tmp_name = f"{DOWNLOAD_CACHE_PATH}{i}"
    cdef unicode txt_name = tmp_name + '.txt'
    cdef bytes byte_line = b""
    cdef unsigned long long total = 0
    cdef int idx = 0
    print(f"jsonl {i:2d} sleep")
    if check_files([tmp_name + '.txt']):
        print(f"jsonl {i:2d} no")
        return
    sleep_till_exists(tmp_name + '.jsonl')
    print(f"jsonl {i:2d} start")

    lock.acquire()
    with open(txt_name + '.tmp', 'w') as o:
        o.write('')
    with open(tmp_name + '.jsonl', 'rb', 2 ** 20) as f:
        with open(txt_name + '.tmp', 'a', 2 ** 20) as o:
            for idx, byte_line in enumerate(f):
                o.write(fix_string(byte_line, i, i, idx, &total) + chr(4))
    lock.release()
    os.rename(txt_name + '.tmp', txt_name)
    if REMOVE_INTERMEDIATE:
        os.remove(tmp_name + '.jsonl')


cpdef void main():
    cdef tuple procs = tuple()
    for path in ('', 'download', 'log', 'done'):
        if not os.path.exists(BASE_PATH + path):
            os.mkdir(BASE_PATH + path)
    if not os.path.exists(DOWNLOAD_CACHE_PATH):
        os.mkdir(DOWNLOAD_CACHE_PATH)

    cdef unicode split_chars = string.digits + " \t\n\r\x0b\x0c"
    for c in string.punctuation:
        split_chars += '\\' + c
    regex = Regex(f"""[{split_chars}]|[^{split_chars}]+""")
    tokenizer = Tokenizer(BPE(unk_token='\x01', cache_capacity=CACHE_CAPACITY, merges=None, dropout=None))
    tokenizer.pre_tokenizer = Split(regex, 'isolated')
    cdef list formatted = [f"{DOWNLOAD_CACHE_PATH}{i}.txt" for i in range(SPLITS)]
    trainer = BpeTrainer(special_tokens=[chr(i) for i in range(256)], vocab_size=VOCAB_SIZE)
    manager = multiprocessing.Manager()
    down_lock = manager.Semaphore(2)
    cdef unsigned short files_exist = 1
    cdef unicode file = ""
    for file in formatted:
        files_exist &= not os.path.exists(file)
    if STREAM and not files_exist:
        queue = manager.Queue(PREFETCH)

        procs = tuple([multiprocessing.Process(target=file_generator, args=(queue, down_lock, i))
                       for i in range(PROCESSES)])
        for p in procs:
            p.start()

        while queue.qsize() < PREFETCH // 2:
            time.sleep(5)

        tokenizer.train_from_iterator(iterator(queue, procs), trainer)

        for p in procs:
            p.join()
    else:
        extract_lock = manager.Semaphore(PROCESSES)
        txt_lock = manager.Semaphore(PROCESSES)

        procs = tuple([multiprocessing.Process(target=download, args=(i, i, down_lock)) for i in range(SPLITS)] +
                      [multiprocessing.Process(target=extract, args=(i, extract_lock)) for i in range(SPLITS)] +
                      [multiprocessing.Process(target=jsonl_to_txt, args=(i, txt_lock)) for i in range(SPLITS)])
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        tokenizer.train(formatted, trainer)

        if REMOVE_LAST_INTERMEDIATE:
            for file in formatted:
                os.remove(file)
    tokenizer.save(".tmp.json")

    with open("tokenizer.json", 'w', errors='ignore') as w, open(".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))

    os.remove(".tmp.json")
