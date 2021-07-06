import typing
import multiprocessing
import uvicorn
from fastapi import FastAPI
from .dataclass import ModelParameter
from .interface import InterfaceWrapper

FUNCTIONS = {}


def api(fn: typing.Callable):
    FUNCTIONS[fn.__name__] = fn
    return fn


@api
async def completion(model_name: str, prompt: str = "", max_tokens: int = 16, temperature: float = 1.,
                     return_probabilities: bool = False):
    return


def main(args):
    app = FastAPI()
    for key, fn in FUNCTIONS.items():
        app.get(key)(fn)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info', workers=args.workers)