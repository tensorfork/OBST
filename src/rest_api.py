import multiprocessing

import uvicorn
from fastapi import FastAPI
from transformers import GPT2TokenizerFast

from .dataclass import ModelParameter
from .interface import InterfaceWrapper


class RestAPI:
    def __init__(self, params: ModelParameter):
        self._functions = {}
        self._interface = InterfaceWrapper(params)
        self._params = params
        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    async def tokenize(self, prompt: str):
        return list(prompt.encode()) if self._params.vocab_size == 256 else self._tokenizer.encode(prompt)

    async def completion(self, prompt: str = "", max_tokens: int = 16, temperature: float = 1.):
        prompt = await self._tokenizer.encode(prompt)
        return self._interface.complete(prompt, temperature, max_tokens)


def get_api_input_and_output_fn(params: ModelParameter):
    rest_api = RestAPI(params)
    fast_api = FastAPI()

    for key, fn in dir(rest_api).items():
        if key.startswith('_') or key.endswith('_'):
            continue
        fast_api.get(key)(fn)

    run = multiprocessing.Process(target=uvicorn.run, daemon=True, args=(fast_api,),
                                  kwargs={'host': '0.0.0.0', 'port': 62220, 'log_level': 'info',
                                          'workers': params.web_workers})
    run.start()

    return rest_api._interface.input_query, rest_api._interface.output_responds
