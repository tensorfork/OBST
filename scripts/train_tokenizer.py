import argparse
import io
import multiprocessing
import os
import shutil
import string

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


# import ftfy

# tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

def file_generator(pid, procs):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = f".tmp.download.{pid}"
    idx = [0]

    if not os.path.exists('pile'):
        os.mkdir('pile')

    def _json_parser(x):
        try:
            return parse_fn(x).as_dict()
        except ValueError:
            return x

    def _write(text):
        with open(f'pile/{pid}_{idx[0]}.txt') as f:
            f.write(ftfy.fix_text(text).replace('    ', '\t'))
        idx[0] += 1

    for i in range(pid, splits, procs):
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with open(tmp_name, 'rb') as f:
            for item in jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)),
                                         loads=_json_parser):
                if isinstance(item, dict):
                    item = item['text']
                if isinstance(item, list):
                    for itm in item:
                        _write(itm)
                else:
                    _write(item)
        os.remove(tmp_name)


def main():
    # setup cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='tokenizer', help="Path for tokenizer dump")
    parser.add_argument("--fixed_file_prefix", type=str, default='fixed-', help="Prefix appended to fixed files")
    parser.add_argument("--files", type=str, default='',
                        help="List of files to train tokenizer on, split by comma")
    parser.add_argument("--pile", type=bool, default=False, help="Whether to download the pile")
    parser.add_argument("--procs", type=int, default=8,
                        help="Number of processes used in pile download. Only used in pile download.")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Items in vocabulary")
    parser.add_argument("--separator", type=int, default=4, help="Separator character")
    parser.add_argument("--cache_capacity", type=int, default=1024 * 1024 * 1024,
                        help="Number of words to keep in BPE cache")
    args = parser.parse_args()

    backslash = "\\"
    chars = f'{string.digits} \t\n\r\x0b\x0c{"".join(f"{backslash}{c}" for c in string.punctuation)}'
    regex = Regex(f"""[{chars}]|[^{chars}]+""")
    special_tokens = [chr(i) for i in range(256)]

    # grab files
    files = args.files.split(',')
    if args.pile:
        if files:
            print("Ignoring --files flag, as --pile is used")
        if not os.path.exists('pile'):
            procs = [multiprocessing.Process(target=file_generator, args=(i, args.procs)) for i in range(args.procs)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
        files = list(os.listdir('pile'))
    for f in files:
        if os.path.exists(args.fixed_file_prefix + f):
            continue
        with open(f, 'r', errors='ignore') as r, open(f'fixed-{f}', 'w', errors='ignore') as w:
            w.write(ftfy.fix_file(r))

    # set up tokenizer
    tokenizer = Tokenizer(BPE(unk_token='\x01', cache_capacity=args.cache_capacity, merges=None, dropout=None))
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=args.vocab_size)
    tokenizer.pre_tokenizer = Split(regex, 'isolated')

    # train and save
    tokenizer.train([args.fixed_file_prefix + f for f in files], trainer)
    tokenizer.save(f".tmp.json")
    with open(f"{args.output}.json", 'w', errors='ignore') as w, open(f".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))


if __name__ == "__main__":
    main()
