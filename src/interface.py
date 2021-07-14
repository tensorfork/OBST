import multiprocessing
import random
import time
import typing

import numpy as np
from transformers import GPT2TokenizerFast

from .dataclass import ModelParameter
from .utils_core import chunks, color_print


def render_video(model_output: typing.List[typing.Tuple[np.ndarray, typing.List[str]]],
                 count: int,
                 params: ModelParameter,
                 save_prefix: str = "",
                 upscale: int = 4,
                 line_split: int = 2,
                 text_color: typing.Tuple[int, int, int] = (255, 0, 255),
                 text_pos: typing.Tuple[int, int] = (10, 625),
                 text_size: float = 1.27,
                 text_thickness: int = 3,
                 text_line_offset: int = 50,
                 prompt_sample_color: typing.Tuple[int, int, int] = (0, 128, 255),
                 prompt_sample_pos: typing.Tuple[int, int] = (50, 50),
                 ):
    import cv2
    writer = cv2.VideoWriter(f"{save_prefix}_{count}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,
                             (params.frame_width * upscale * len(model_output), params.frame_height * upscale))

    for idx in range(len(model_output[0][0])):
        frame = []
        for sub_idx in range(len(model_output)):

            sub_frame = model_output[sub_idx][0][idx]
            if params.use_discrete_video_loss:
                sub_frame = sub_frame * (256 / params.color_quantization_value)
            else:
                sub_frame = sub_frame * (params.color_quantization_value - 1)
            import scipy.ndimage
            sub_frame = scipy.ndimage.zoom(sub_frame, (upscale, upscale, 1), order=0)
            sub_frame = np.uint8(sub_frame)
            cv2.cvtColor(sub_frame, cv2.COLOR_RGB2BGR)

            text = model_output[sub_idx][1]
            if text is not None:
                for i, _text in enumerate(chunks(text[idx], params.language_token_per_frame // line_split)):
                    cv2.putText(sub_frame, _text, (text_pos[0], text_pos[1] + text_line_offset * i),
                                cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

            if params.use_autoregressive_sampling:
                prompt_sample_text = 'prompt' if idx < params.initial_autoregressive_position else 'sample'
                cv2.putText(sub_frame, prompt_sample_text, prompt_sample_pos, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, prompt_sample_color, text_thickness)

            frame.append(sub_frame)

        frame = np.concatenate(frame, axis=1)
        writer.write(frame)

    writer.release()


def process_token_output(token_out: np.ndarray, padding_token: int = -1, do_argmax: bool = True,
                         bpe_tokenizer: GPT2TokenizerFast = None) -> typing.List[str]:
    _shape = token_out.shape
    if do_argmax:
        voc_size = _shape[3] * _shape[4] if len(_shape) > 4 else _shape[3]
        token_out = np.reshape(token_out, newshape=(_shape[0], _shape[1] * _shape[2], voc_size))
        token_out = np.argmax(token_out, axis=2)
    else:
        token_out = np.reshape(token_out, newshape=(_shape[0], _shape[1] * _shape[2]))

    token_out_str = []

    for token in token_out:
        if padding_token > -1 and padding_token in token:
            token = token[:token.tolist().index(padding_token)]

        if bpe_tokenizer is None:
            token_out_str.append(
                "".join(
                    chr(tok) if tok > 31 and tok != 127 and tok != 10 else " "
                    for tok in token
                )
            )

        else:
            token_out_str.append(bpe_tokenizer.decode([int(tok) for tok in token]))

    return token_out_str


def process_video_output(out_frame: np.ndarray, params: ModelParameter) -> np.ndarray:
    out_frame = np.reshape(out_frame, (params.time_patch_size, params.frame_height_patch, params.frame_width_patch,
                                       params.time_patch, params.patch_size, params.patch_size, params.color_channels))

    out_frame = np.transpose(out_frame, [0, 3, 1, 4, 2, 5, 6])
    out_frame = np.reshape(out_frame, (params.n_ctx, params.frame_height, params.frame_width, 3))

    return out_frame


def gen_sample_fn(params: ModelParameter):
    state = {'sample_index': 0}

    def _video_fn(out):
        print('sample_idx:', state['sample_index'])

        token_inp = None
        token_out = None
        render_input = []

        frame_out = out[0][0]
        if params.use_autoregressive_sampling:
            frame_out = frame_out[:-1]

        frame_out = process_video_output(frame_out, params)

        if params.use_language:
            token_out = process_token_output(out[2][0], params.padding_token, not params.use_autoregressive_sampling)

        if not params.use_autoregressive_sampling:
            frame_inp = out[1][0]
            frame_inp = frame_inp[1:params.time_patch_size + 1]
            frame_inp = process_video_output(frame_inp, params)

            if params.use_language:
                token_inp = process_token_output(out[3][0], params.padding_token, False)

            render_input.append((frame_inp, token_inp))

        render_input.append((frame_out, token_out))

        render_video(render_input, state['sample_index'], params)

        state['sample_index'] += 1
        if state['sample_index'] >= params.num_of_sample:
            exit()

    def _text_fn(out):
        print('sample_idx:', state['sample_index'])

        bpe_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') if params.vocab_size != 256 and \
                                                                     params.vocab_size > 256 else None

        if params.use_autoregressive_sampling:

            if params.debug_sample:
                score = np.int(np.mean(np.equal(out[0][0], out[0][1])) * 100)
                print(f"similarity score: {score}%\n")
                print([process_token_output(out[0], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0]])
                print([process_token_output(out[0], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[1]])
                print([process_token_output(out[1], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0]])
                print('')

            print('\n------\n')
            color_print(params, 'Prompt:')
            assert params.initial_autoregressive_position > 0
            print(process_token_output(out[1][:, :params.initial_autoregressive_position - 1], do_argmax=False,
                                       bpe_tokenizer=bpe_tokenizer)[0])
            color_print(params, 'Output:')
            print(process_token_output(out[0][:, params.initial_autoregressive_position:], do_argmax=False,
                                       bpe_tokenizer=bpe_tokenizer)[0].rstrip())
        else:
            print('target:')
            print(process_token_output(out[1], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0])
            print('\nsample:')
            print(process_token_output(out[0], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0])

        state['sample_index'] += 1
        if state['sample_index'] >= params.num_of_sample:
            exit()

        print('\n')

    return _video_fn if params.model_mode == 'jannet' else _text_fn


def get_command_line_input_and_output_fn(params: ModelParameter):
    bpe_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') if params.vocab_size != 256 and \
                                                                 params.vocab_size > 256 else None

    samp_temp = params.sampling_temperature
    end_iter = params.n_ctx
    _iter_pos = [0]
    _end_iter = [0]

    def input_fns():

        while True:
            color_print(params, 'Enter Quary:')
            query = input()

            if bpe_tokenizer is None:
                query = [ord(q) for q in query]
            else:
                query = bpe_tokenizer.encode(query)

            if len(query) >= params.n_ctx:
                color_print(params, f'Query is to long, the maximum number tokens is '
                                    f'{params.n_ctx}, but you have {len(query)} tokens.')
                continue

            iter_pos = len(query) + 1
            _iter_pos[0] = iter_pos

            _end_iter[0] = end_iter

            query = query + [0] * (params.n_ctx - len(query))
            query = np.reshape(np.array(query, np.int32), newshape=(1, params.n_ctx, 1))
            break

        return query, np.array([iter_pos], np.int32), \
               np.array([samp_temp], np.float32), np.array([end_iter], np.int32)

    def output_fn(out):
        color_print(params, 'Responds:')
        print(process_token_output(out[0][:, _iter_pos[0]:][:, :_end_iter[0]], do_argmax=False,
                                   bpe_tokenizer=bpe_tokenizer)[0].rstrip())
        print('')

    return input_fns, output_fn


class InterfaceWrapper:
    def __init__(self, params: ModelParameter):
        self.params = params
        self.manager = multiprocessing.Manager()
        self.input_prompt_id = self.manager.Value(int, 0)
        self.tpu_input_id = self.manager.Value(int, 0)
        self.output_prompt_id = self.manager.Value(int, 0)
        self.input = self.manager.dict()
        self.output = self.manager.dict()

    def blocked_get(self, inp: dict, key: int):
        while key not in inp:
            time.sleep(self.params.default_sleep_duration)
        return inp.pop(key)

    def increment(self, idx) -> int:
        prompt_id = idx.get()
        idx.set(prompt_id + 1)
        return prompt_id

    def complete(self, query: typing.List[int], temperature: float, response_len: int, debug: bool = False,
                 asynchronous: bool = False) -> typing.Union[typing.Callable, typing.Tuple[np.array, np.array],
                                                             np.array]:
        iter_pos = len(query)

        if iter_pos >= self.params.n_ctx or max(query) >= self.params.vocab_size:
            return None

        prompt_id = self.increment(self.input_prompt_id)

        query = query + [random.randint(0, self.params.vocab_size - 1) for _ in range((self.params.n_ctx - len(query)))]
        query = np.reshape(np.array(query, np.int32), newshape=(1, self.params.n_ctx, 1))

        self.input[prompt_id] = (query, np.array([iter_pos], np.int32), np.array([temperature], np.float32),
                                 np.array([min(response_len + len(query), self.params.n_ctx)], np.int32))

        def _result():
            response = self.blocked_get(self.output, prompt_id)[0].astype(np.int64)
            out = response[0, iter_pos:].flatten()
            return (out, response) if debug else out

        return _result if asynchronous else _result()

    def input_query(self):
        return self.blocked_get(self.input, self.increment(self.tpu_input_id))

    def output_responds(self, out):
        self.output[self.increment(self.output_prompt_id)] = out


def get_similarity_input_and_output_fn(params: ModelParameter):
    interface = InterfaceWrapper(params)

    def run():
        time.sleep(10)

        for idx in range(params.num_of_sample):
            query = [random.randint(0, params.vocab_size - 1) for _ in range(min(32, params.n_ctx - 8))]

            out = [interface.complete(query=query, temperature=0.0, response_len=params.n_ctx, debug=True,
                                      asynchronous=True) for _ in range(params.equal_debugging_items_per_check)]
            base, *out = [f() for f in out]

            score = float(np.mean([np.mean(np.equal(base, o)) * 100 for o in out]))
            print(f"test:{idx} similarity score: {score:6.2f}%\n")

    run = multiprocessing.Process(target=run, daemon=True)
    run.start()

    return interface.input_query, interface.output_responds
