import multiprocessing
import typing

import uvicorn
from fastapi import FastAPI
from transformers import GPT2TokenizerFast

from .dataclass import ModelParameter
from .interface import InterfaceWrapper


class RestAPI:
    def __init__(self, params: ModelParameter):
        self.functions = {}
        self.interface = InterfaceWrapper(params)
        self.params = params
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    async def tokenize(self, prompt: str):
        return list(prompt.encode()) if self.params.vocab_size == 256 else self.tokenizer.encode(prompt)

    async def completion(self, prompt: str = "", max_tokens: int = 16, temperature: float = 1.):
        prompt = await self.tokenizer.encode(prompt)
        return self.interface.complete(prompt, temperature, max_tokens)


def get_api_input_and_output_fn(params: ModelParameter):
    rest_api = RestAPI(params)
    fast_api = FastAPI()

    for key, fn in rest_api.__dict__.items():
        if key.startswith('_') or key.endswith('_') or not isinstance(fn, typing.Callable):
            continue
        fast_api.get(key)(fn)

    run = multiprocessing.Process(target=uvicorn.run, daemon=True, args=(fast_api,),
                                  kwargs={'host': '0.0.0.0', 'port': 62220, 'log_level': 'info',
                                          'workers': params.web_workers})
    run.start()

    return rest_api.interface.input_query, rest_api.interface.output_responds
