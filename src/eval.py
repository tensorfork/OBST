import typing

import cv2
import numpy as np
import scipy.ndimage
from transformers import GPT2TokenizerFast

from src.dataclass import ModelParameter
from src.utils_core import chunks


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
    writer = cv2.VideoWriter(f"{save_prefix}_{count}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 1,
                             (params.frame_width * upscale * len(model_output), params.frame_height * upscale))

    for idx in range(len(model_output[0][0])):
        frame = []
        for sub_idx in range(len(model_output)):

            sub_frame = model_output[sub_idx][0][idx]
            sub_frame = sub_frame * 255
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
        token_out = np.reshape(token_out, newshape=(_shape[0], _shape[1] * _shape[2], _shape[3]))
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

        bpe_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') if params.vocab_size != 256 and\
                                                                     params.vocab_size > 256 else None

        if params.use_autoregressive_sampling:

            if params.debug_sample:
                score = np.int(np.mean(np.equal(out[0][0], out[0][1])) * 100)
                print(f"similarity score: {score}%\n")
                #print([process_token_output(out[0], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0]])
                #print([process_token_output(out[0], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[1]])
                #print([process_token_output(out[1], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0]])
                #print('')

            else:

                print('Prompt:')
                print(process_token_output(out[1], do_argmax=False,
                                           bpe_tokenizer=bpe_tokenizer)[0][:params.initial_autoregressive_position])
                print('\noutput:')
                print(process_token_output(out[0], do_argmax=False,
                                           bpe_tokenizer=bpe_tokenizer)[0][params.initial_autoregressive_position:])
        else:
            print('target:')
            print(process_token_output(out[1], do_argmax=False, bpe_tokenizer=bpe_tokenizer)[0])
            print('\nsample:')
            print(process_token_output(out[0], do_argmax=True, bpe_tokenizer=bpe_tokenizer)[0])

        state['sample_index'] += 1
        if state['sample_index'] >= params.num_of_sample:
            exit()

        #print('\n')

    return _video_fn if params.model_mode == 'jannet' else _text_fn
