import argparse
import io
import os
import shutil
import string

import ftfy
import jsonlines
import time
import jsonpickle
import requests
import simdjson
import zstandard
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer


def file_generator(pid, procs):
    base_url = 'http://eaidata.bmk.sh/data/pile/train/%s.jsonl.zst'
    splits = 30
    parse_fn = simdjson.Parser().parse
    tmp_name = f".tmp.download.{pid}"
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
        with requests.get(base_url.replace("%s", str(i).zfill(2)), stream=True) as r, open(tmp_name, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        with open(tmp_name, 'rb') as f:
            read = jsonlines.Reader(io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f)), loads=_parser)
            for idx, item in enumerate(read):
                if isinstance(item, dict):
                    item = item['text']
                if isinstance(item, list):
                    for itm in item:
                        yield _write(itm)
                else:
                    yield _write(item)
                if idx % 100000:
                    print(f"Chars: {total[0]} - Took: {time.time():.1f}s")
        print(f"FINISHED {i}")
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
    tokenizer.train_from_iterator(file_generator(0, 1), trainer)
    tokenizer.save(f".tmp.json")
    with open(f"{args.output}.json", 'w', errors='ignore') as w, open(f".tmp.json", 'r', errors='ignore') as r:
        w.write(jsonpickle.dumps(jsonpickle.loads(r.read()), indent=4))


if __name__ == "__main__":
    main()
