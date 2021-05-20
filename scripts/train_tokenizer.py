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


def file_generator(queue: Queue, lock: threading.Semaphore, pid, procs):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = f"E:\\Pile\\download.{pid}.zstd"
    total = [0]

    if not os.path.exists('pile'):
        os.mkdir('pile')

    def _parser(x):
        try:
            return parse_fn(x).as_dict()
        except:
            return x

    def _write(text):
        out = ftfy.fix_text(text).replace('    ', '\t')
        total[0] += len(out)
        return out

    for i in range(pid, splits, procs):
        total[0] = 0
        print(f"Starting {i} at {datetime.datetime.now()}")
        lock.acquire()
        start = time.time()
        print(f"Downloading {i}")
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print(f"Finished downloading {i} after {time.time() - start:.1f}s")
        lock.release()
        start = time.time()
        with open(tmp_name, 'rb') as f:
            read = jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)), loads=_parser)
            for idx, item in enumerate(read):
                if isinstance(item, dict):
                    item = item['text']
                if isinstance(item, list):
                    for itm in item:
                        queue.put(_write(itm))
                else:
                    queue.put(_write(item))
                if idx % 100000 == 0:
                    print(f"Slice: {i} - Chars: {total[0]} - Took: {time.time() - start:.1f}s")
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
    parser.add_argument("--separator", type=int, default=4, help="Separator character")
    parser.add_argument("--cache_capacity", type=int, default=1024 * 1024 * 1024,
                        help="Number of words to keep in BPE cache")
    args = parser.parse_args()

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
    queue = manger.Queue(16)
    lock = manger.Semaphore()

    procs = [multiprocessing.Process(target=file_generator, args=(queue, lock, i, args.procs))
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

    while queue.qsize() < 4:
        time.sleep(5)
    tokenizer.train_from_iterator(_iterator(), trainer)
    for p in procs:
        p.join()

    tokenizer.save(f".tmp.json")

    with open(f"{args.output}.json", 'w', errors='ignore') as w, open(f".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))


if __name__ == "__main__":
    main()
