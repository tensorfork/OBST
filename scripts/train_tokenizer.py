import argparse
import datetime
import io
import multiprocessing
import os
import shutil
import string
import threading
import time
from queue import Queue

import ftfy
import jsonlines
import jsonpickle
import requests
import simdjson
import zstandard
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer


def file_generator(queue: Queue, lock: threading.Semaphore, pid: int, procs: int, base_path: str):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = f"{base_path}download\\{pid}.zstd"
    log_path = f"{base_path}log\\{pid}.txt"
    total = [0]

    with open(log_path, 'w') as f:
        f.write('')

    def _parser(x):
        return parse_fn(x.encode()).as_dict()

    def _write(text):
        out = ftfy.fix_text(text).replace('    ', '\t')
        total[0] += len(out)
        return out

    for i in range(pid, splits, procs):
        i = 2

        def _log(text):
            with open(log_path, 'a') as f:
                f.write(f'Proc: {pid} | Slice: {i} | Time: {datetime.datetime.now()} | {text}\n')

        total[0] = 0
        _log(f"Starting")
        lock.acquire()
        _log(f"Downloading")
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        _log(f"Finished downloading")
        lock.release()
        with open(tmp_name, 'rb') as f:
            read = jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)), loads=_parser)
            for idx, item in enumerate(read):
                item = item['text']
                if isinstance(item, list):
                    for itm in item:
                        queue.put(_write(itm))
                else:
                    queue.put(_write(item))
                if idx % 100000 == 0:
                    _log(f"{total[0] / 2 ** 20:9.2f}MB")
        os.remove(tmp_name)


def main():
    # setup cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='tokenizer', help="Path for tokenizer dump")
    parser.add_argument("--fixed_file_prefix", type=str, default='fixed-', help="Prefix appended to fixed files")
    parser.add_argument("--files", type=str, default='',
                        help="List of files to train tokenizer on, split by comma")
    parser.add_argument("--pile", type=bool, default=False, help="Whether to download the pile")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Items in vocabulary")
    parser.add_argument("--procs", type=int, default=8, help="Number of processes")
    parser.add_argument("--prefetch", type=int, default=128, help="Prefetch queue size")
    parser.add_argument("--base_path", type=str, default='E:\\Pile\\', help="Path of data and logs")
    parser.add_argument("--separator", type=int, default=4, help="Separator character")
    parser.add_argument("--cache_capacity", type=int, default=1024 * 1024 * 1024,
                        help="Number of words to keep in BPE cache")
    args = parser.parse_args()

    for path in ('', 'download', 'log'):
        if not os.path.exists(args.base_path + path):
            os.mkdir(args.base_path + path)

    backslash = "\\"
    chars = f'{string.digits} \t\n\r\x0b\x0c{"".join(f"{backslash}{c}" for c in string.punctuation)}'
    regex = Regex(f"""[{chars}]|[^{chars}]+""")
    special_tokens = [chr(i) for i in range(256)]

    # set up tokenizer
    tokenizer = Tokenizer(BPE(unk_token='\x01', cache_capacity=args.cache_capacity, merges=None, dropout=None))
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=args.vocab_size)
    tokenizer.pre_tokenizer = Split(regex, 'isolated')

    # train and save
    manger = multiprocessing.Manager()
    queue = manger.Queue(args.prefetch)
    lock = manger.Semaphore()

    procs = [multiprocessing.Process(target=file_generator, args=(queue, lock, i, args.procs, args.base_path))
             for i in range(args.procs)]
    for p in procs:
        p.start()

    def _iterator():
        while True:
            try:
                yield queue.get(timeout=60)
            except:
                if not any(p.is_alive() for p in procs):
                    return

    while queue.qsize() < args.prefetch // 2:
        time.sleep(5)
    tokenizer.train_from_iterator(_iterator(), trainer)
    for p in procs:
        p.join()

    tokenizer.save(f".tmp.json")

    with open(f"{args.output}.json", 'w', errors='ignore') as w, open(f".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))


if __name__ == "__main__":
    main()
