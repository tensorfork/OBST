import multiprocessing
import typing

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2TokenizerFast

from .dataclass import ModelParameter
from .interface import InterfaceWrapper


class Tokens(BaseModel):
    tokens: typing.List[int]


class TokenCompletion(BaseModel):
    token_completion: typing.List[int]


class Completion(BaseModel):
    completion: str


class RestAPI:
    def __init__(self, params: ModelParameter):
        self._functions = {}
        self._interface = InterfaceWrapper(params)
        self._params = params
        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    async def tokenize(self, prompt: str) -> Tokens:
        out = list(prompt.encode()) if self._params.vocab_size == 256 else self._tokenizer.encode(prompt)
        return Tokens(tokens=out)

    async def token_completion(self, prompt: str = "", max_tokens: int = 16,
                               temperature: float = 1.) -> TokenCompletion:
        out = self._interface.complete((await self.tokenize(prompt)).tokens, temperature, max_tokens).tolist()
        return TokenCompletion(token_completion=out)

    async def completion(self, prompt: str = "", max_tokens: int = 16, temperature: float = 1.) -> Completion:
        out = self._tokenizer.decode((await self.token_completion(prompt, max_tokens, temperature)).token_completion)
        return Completion(completion=out)


def get_api_input_and_output_fn(params: ModelParameter):
    rest_api = RestAPI(params)
    fast_api = FastAPI()

    for key in dir(rest_api):
        if key.startswith('_') or key.endswith('_'):
            continue
        fast_api.get('/' + key)(getattr(rest_api, key))

    run = multiprocessing.Process(target=uvicorn.run, daemon=True, args=(fast_api,),
                                  kwargs={'host': '0.0.0.0', 'port': 62220, 'log_level': 'info',
                                          'workers': params.web_workers})
    run.start()

    return rest_api._interface.input_query, rest_api._interface.output_responds
