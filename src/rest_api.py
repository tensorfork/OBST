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

    def api(self, fn: typing.Callable):
        self.functions[fn.__name__] = fn

    @api
    async def tokenize(self, prompt: str):
        return list(prompt.encode()) if self.params.vocab_size == 256 else self.tokenizer.encode(prompt)

    @api
    async def completion(self, prompt: str = "", max_tokens: int = 16, temperature: float = 1.):
        prompt = await self.tokenizer.encode(prompt)
        return self.interface.complete(prompt, temperature, max_tokens)

    def main(self):
        app = FastAPI()

        for key, fn in self.functions.items():
            app.get(key)(fn)

        uvicorn.run(app, host='0.0.0.0', port=62220, log_level='info', workers=self.params.web_workers)


def get_api_input_and_output_fn(params: ModelParameter):
    api = RestAPI(params)

    run = multiprocessing.Process(target=api.main, daemon=True)
    run.start()

    return api.interface.input_query, api.interface.output_responds
