import multiprocessing
import typing

import uvicorn
from fastapi import FastAPI, HTTPException
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


class SanitizedTokens(BaseModel):
    tokens: typing.List[int]


class CompletionInput(BaseModel):
    prompt: str = ""
    max_tokens: int = 16
    temperature: float = 1.
    error: bool = True


class RestAPI:
    def __init__(self, params: ModelParameter):
        self._interface = InterfaceWrapper(params)
        self._params = params
        self._tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    async def check_tokens(self, tokens: typing.List[int], error: bool = True) -> SanitizedTokens:
        if tokens and max(tokens) > self._params.vocab_size:
            if error:
                raise HTTPException(status_code=400, detail=f"Invalid tokens sent. Tokens go up to "
                                                            f"{self._params.vocab_size} but received {max(tokens)}.")
            tokens = [t for t in tokens if t < self._params.vocab_size]
        if len(tokens) > self._params.n_ctx:
            if error:
                raise HTTPException(status_code=400, detail=f"Context too big. The model supports up to "
                                                            f"{self._params.n_ctx} tokens but received {len(tokens)}.")
            tokens = tokens[:self._params.n_ctx]
        return SanitizedTokens(tokens=tokens)

    async def encode(self, prompt: str) -> Tokens:
        out = list(prompt.encode()) if self._params.vocab_size == 256 else self._tokenizer.encode(prompt)
        return Tokens(tokens=out)

    async def decode(self, prompt: typing.List[int]) -> Completion:
        out = ''.join(chr(c) for c in prompt) if self._params.vocab_size == 256 else self._tokenizer.encode(prompt)
        return Completion(completion=out)

    async def token_completion(self, params: CompletionInput) -> TokenCompletion:
        tokens = (await self.encode(params.prompt)).tokens
        tokens = (await self.check_tokens(tokens, params.error)).tokens
        out = self._interface.complete(tokens, params.temperature, len(tokens) + params.max_tokens)
        out = out.tolist()[:params.max_tokens]
        return TokenCompletion(token_completion=out)

    async def completion(self, params: CompletionInput) -> Completion:
        return await self.decode((await self.token_completion(params)).token_completion)


def get_api_input_and_output_fn(params: ModelParameter):
    rest_api = RestAPI(params)
    fast_api = FastAPI()

    for key in dir(rest_api):
        if key.startswith('_') or key.endswith('_'):
            continue
        fn = getattr(rest_api, key)
        fast_api.post('/' + key, response_model=typing.get_type_hints(fn)["return"])(fn)

    run = multiprocessing.Process(target=uvicorn.run, daemon=True, args=(fast_api,),
                                  kwargs={'host': '0.0.0.0', 'port': 62220, 'log_level': 'info',
                                          'workers': params.web_workers})
    run.start()

    return rest_api._interface.input_query, rest_api._interface.output_responds
