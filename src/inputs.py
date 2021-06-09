"""
Contains input pipeline code that generates tensorflow datasets if called
"""
import random

import numpy as np
import tensorflow as tf2

from .dataclass import ModelParameter, align_tensor_op

tf = tf2.compat.v1
Dataset = tf2.data.Dataset


def split_files(filenames, slice_index, slice_count, seed, runs_log=None):
    if not filenames:
        raise ValueError
    files = sorted(filenames)
    if seed != 0:
        random.seed(seed)
        random.shuffle(files)

    element_skip = [0] * len(files)

    if runs_log is not None:
        file_list_skip, element_skip = simulate_data_pipeline(runs_log, files)
        files = [files[i] for i, s in enumerate(file_list_skip) if not s]
        element_skip = [element_skip[i] for i, s in enumerate(file_list_skip) if not s]

    return files[slice_index::slice_count], element_skip[slice_index::slice_count]


def simulate_data_pipeline(runs_log, file_list):
    file_list = [int(str(f).split('_')[-1].strip('.tfrecord')) for f in file_list]

    file_list_skip = [False] * len(file_list)
    element_skip = [0] * len(file_list)
    file_idx_list = list(range(len(file_list)))

    for run in runs_log:

        # Remove full skip TF-records.
        _file_list = [file_list[i] for i, s in enumerate(file_list_skip) if not s]
        # Remove TF-record element skips for full skip remove.
        _element_skip = [element_skip[i] for i, s in enumerate(file_list_skip) if not s]
        # Remove TF-record index.
        _file_idx_list = [file_idx_list[i] for i, s in enumerate(file_list_skip) if not s]
        # Remove already used elements.
        _file_list = [_file_list[i] - s for i, s in enumerate(_element_skip)]

        slice_count = run['slice_count']
        ctx = run['ctx']
        step_stop_count = run['steps'] * run['grad_accumulation'] * (run['batch_size'] // slice_count)
        interleave_size = run['interleave_size']
        token_patch_size = run['token_patch_size']

        for slice_index in range(slice_count):
            _file_list_slice = _file_list[slice_index::slice_count]
            _file_idx_list_slice = _file_idx_list[slice_index::slice_count]

            _step_stop_count = step_stop_count

            for interleave_idx in range(0, len(_file_list_slice), interleave_size):
                # Remove all elements for a TF-record how not fit in to the last sample.
                # Also remove patch_size element for the X Y sample split.
                interleave_chunk = [c - ((c - token_patch_size) % ctx) - token_patch_size for c in
                                    _file_list_slice[interleave_idx:interleave_idx + interleave_size]]
                _interleave_chunk = interleave_chunk.copy()

                sum_step_in_interleave = sum(interleave_chunk) // ctx
                if sum_step_in_interleave > _step_stop_count:

                    # Take from interleave until all TF-records in interleave batch are emty
                    # or until step_stop_count is done.
                    inter_idx = 0
                    while sum(interleave_chunk) > 0 and _step_stop_count > 0:

                        # Jump the interleave sample pos to a non depleted TF-record.
                        while interleave_chunk[inter_idx] <= 0:
                            inter_idx += 1
                            # Set the Interleave index back to zero when it has reached the end of the interleave batch.
                            if inter_idx >= len(interleave_chunk):
                                inter_idx = 0

                        # Remove n ctx elements from the TF-record.
                        interleave_chunk[inter_idx] = interleave_chunk[inter_idx] - ctx
                        _step_stop_count -= 1

                        inter_idx += 1
                        # Set the Interleave index back to zero when it has reached the end of the interleave batch.
                        if inter_idx >= len(interleave_chunk):
                            inter_idx = 0

                    # Calculate how much of a TF-record has been used.
                    remove = [_inter - inter for _inter, inter in zip(_interleave_chunk, interleave_chunk)]

                    # Find out the actual index of the TF-record and then add the update the element skip for the tfrecord.
                    # And also determine if one of the used TF-record has been depleted and update the file skip flack.
                    for c_i in range(len(interleave_chunk)):
                        file_idx = _file_idx_list_slice[interleave_idx + c_i]
                        if interleave_chunk[c_i] <= 0:
                            file_list_skip[file_idx] = True
                        element_skip[file_idx] = element_skip[file_idx] + remove[c_i]

                    # Brake the interleave sample loop if step_stop_count is done.
                    if step_stop_count <= 0:
                        break

                else:

                    _step_stop_count -= sum_step_in_interleave
                    for c_i in range(len(interleave_chunk)):
                        file_idx = _file_idx_list_slice[interleave_idx + c_i]
                        file_list_skip[file_idx] = True
                        element_skip[file_idx] = _interleave_chunk[c_i]

        # Only skip full tfrecord when there all TF-records in one interleave batch are depleted.
        for slice_index in range(slice_count):
            file_list_skip_slice = file_list_skip[slice_index::slice_count]
            file_idx_list_slice = file_idx_list[slice_index::slice_count]

            for interleave_idx in range(0, len(file_list_skip_slice), interleave_size):
                full_depleted = sum(
                        file_list_skip_slice[interleave_idx:interleave_idx + interleave_size]) == interleave_size
                for idx in file_idx_list_slice[interleave_idx:interleave_idx + interleave_size]:
                    file_list_skip[idx] = full_depleted

    return file_list_skip, element_skip


def get_video_decoder(params, language_token_num_per_frame=0, frame_height=None, frame_width=None, color_channels=None,
                      color_quantization_value=256):
    '''
    :param language_token_num_per_frame: The number of language tokens per single frame.
    If this is 0 (default) language tokens are disabled.
    :param frame_height:
    :param frame_width:
    :param color_channels:

    This function will return a frame decoder function, that can than be used to decode tf.records.
    '''

    decode_language_token = language_token_num_per_frame > 0
    token_range = tf.range(0, language_token_num_per_frame)

    # Decoding Key.
    features = {
            'frame':      tf.FixedLenFeature([], tf.string),
            'concat':     tf.FixedLenFeature([], tf.int64),
            'skip_frame': tf.FixedLenFeature([], tf.int64)
            }

    if decode_language_token:
        features.update({
                'tokens': tf.FixedLenFeature([language_token_num_per_frame], tf.int64),
                'mask':   tf.FixedLenFeature([], tf.int64)
                })

    three_axes = params.three_axes

    color_channels = params.color_channels
    patch_size = params.patch_size
    channel_color_size = params.channel_color_size

    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch

    frame_shape = [frame_height_patch, frame_width_patch]
    if not three_axes:
        frame_shape = [np.prod(frame_shape)]
    frame_shape.append(channel_color_size)

    out_frame_shape = frame_shape.copy()
    if params.use_bit_fold_input_pipeline:
        out_frame_shape.insert(-1, params.fold_count)

    multi = [1]
    for _ in range(params.fold_count - 1):
        multi.append(multi[-1] * (2 ** params.bit_fold_value))

    def op_decod(frame):

        if color_quantization_value != 256:
            frame = tf2.cast(frame, dtype=tf2.float32)
            frame = tf2.round(frame * ((color_quantization_value - 1) / 255))
            frame = tf2.cast(frame, dtype=(tf2.int64 if params.use_bit_fold_input_pipeline else tf2.uint8))

        frame = tf2.reshape(frame, (frame_height_patch, patch_size, frame_width_patch, patch_size, color_channels))
        frame = tf2.transpose(frame, [1, 3, 0, 2, 4])

        frame = tf2.reshape(frame, out_frame_shape)

        if params.use_bit_fold_input_pipeline:
            _multi = tf2.expand_dims(tf2.expand_dims(tf2.constant(multi, dtype=tf2.int64), axis=-1), axis=0)
            frame = tf2.reduce_sum((frame * _multi), axis=-2)
            frame = tf2.cast(frame, tf.uint32)

        return frame

    def frame_decoder(proto):
        '''
        :param proto: Proto buffer to be decoded.
        :return: tensor with decode frame.

        This Function will decode frame from proto buffer.
        '''

        sample = tf.parse_single_example(proto, features)
        concat = sample['concat']
        skip_frame = sample['skip_frame']

        if skip_frame > 0 or concat > 0:
            frame = tf.zeros(shape=frame_shape, dtype=(tf.uint32 if params.use_bit_fold_input_pipeline else tf.uint8))
        else:
            frame = tf.image.decode_image(sample['frame'])
            frame = op_decod(frame)

        if decode_language_token:
            tokens = sample['tokens']
            mask = sample['skip_frame']

            b_mask = tf.less_equal(token_range, tf.cast(mask, tf.int32))

            return frame, concat, skip_frame, tokens, b_mask

        return frame, concat, skip_frame

    return tf.function(frame_decoder)


def _text_decoder(decoder, data: tf.Tensor, ctx: int, patch_size: int, chunk_size: int,
                  shuffle_buffer: int = 0, _skip=None, parallel_batch: int = 1):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param data: protobuf object to decode
    :param ctx: context size of generated dataset
    :param chunk_size: batch size directly after creating the dataset
    :return: tensorflow dataset of token
    """

    def chunk(tfrecorddataset):
        data = decoder(tfrecorddataset)
        if _skip is not None:
            data = data.skip(tf.cast(_skip, dtype=tf.int64))
        if chunk_size > 0:
            data = data.batch(chunk_size, deterministic=True)
        data = data.window(size=ctx + patch_size, shift=ctx, stride=1, drop_remainder=True)
        data = data.interleave(lambda x: x.batch(ctx + patch_size, drop_remainder=True, deterministic=True),
                               cycle_length=1)
        return data

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk, cycle_length=1)


@tf.function
def decode_bytestring(proto):
    text_slice = tf.parse_single_example(proto, {'text': tf.FixedLenFeature([], tf.string)})['text']
    data = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.strings.unicode_decode(text_slice, 'UTF-8'), (-1, 1)))
    return data


@tf.function
def decode_intstring(proto):
    x = tf.parse_single_example(proto, {'text': tf.VarLenFeature(tf.int64)})
    x = x['text']
    x = tf.sparse.to_dense(x)
    x = tf.cast(x, tf.int32)
    x = tf.data.Dataset.from_tensor_slices(x)
    return x


def dataset_text(path: str, params: ModelParameter, sub_batch_size: int, slice_index, slice_count) -> tf.data.Dataset:
    """
    Creates a text dataset containing shuffled and prefetched windows.
    :param path: Path to dataset (in google cloud bucket)
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    three_axes = params.three_axes

    time_patch = params.time_patch
    token_patch_size = params.token_patch_size
    language_token_patch = params.language_token_patch
    language_token_per_frame = params.language_token_per_frame

    time_patch_size = params.time_patch_size
    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch
    channel_color_size = params.channel_color_size

    assert not (language_token_per_frame > 0 and time_patch > 1), \
        ("Time patch and language token are currently not supported together")

    padding_token = tf.constant([[params.padding_token]] * (time_patch_size + 1), dtype=tf.int32)
    padding_token = tf.data.Dataset.from_tensors(padding_token).repeat()

    if three_axes:
        padding_frame = tf.zeros((time_patch_size + 1, frame_height_patch, frame_width_patch,
                                  channel_color_size), dtype=tf.uint8)
    else:
        padding_frame = tf.zeros((time_patch_size + 1, frame_height_patch * frame_width_patch,
                                  channel_color_size), dtype=tf.uint8)

    padding_frame = tf.data.Dataset.from_tensors(padding_frame).repeat()

    padding_frame_mask = tf.zeros((time_patch_size), dtype=tf.bool)
    padding_frame_mask = tf.data.Dataset.from_tensors(padding_frame_mask).repeat()

    padding_cat_mask = tf.ones((time_patch_size), dtype=tf.bool)
    padding_cat_mask = tf.data.Dataset.from_tensors(padding_cat_mask).repeat()

    def _memory_func(x, _padding_token, _padding_frame, _padding_frame_mask, _padding_cat_mask):

        x = tf.reshape(x, (sub_batch_size, time_patch_size + 1, language_token_per_frame - 1))
        _padding_token = tf.reshape(_padding_token, (sub_batch_size, time_patch_size + 1, 1))
        x = tf.cast(x, tf.int32)
        x = tf.concat([x, _padding_token], axis=2)

        x = tf.reshape(x, (sub_batch_size, time_patch_size + 1, language_token_patch, token_patch_size))

        token_x = x[:, :time_patch_size]
        token_y = x[:, 1:time_patch_size + 1]

        if three_axes:
            _padding_frame = tf.reshape(_padding_frame, (sub_batch_size,
                                                         time_patch_size + 1,
                                                         frame_height_patch,
                                                         frame_width_patch,
                                                         channel_color_size))
        else:
            _padding_frame = tf.reshape(_padding_frame, (sub_batch_size,
                                                         time_patch_size + 1,
                                                         frame_height_patch * frame_width_patch,
                                                         channel_color_size))

        # _padding_token_mask = tf.reshape(_padding_token_mask,
        #                                 (sub_batch_size, time_patch_size, language_token_patch, token_patch_size))

        # _padding_cat_mask = tf.reshape(_padding_cat_mask, (sub_batch_size, time_patch_size))

        _padding_token_mask = tf.not_equal(token_y, tf.constant(params.concat_token, dtype=tf.int32))

        return {'frame':       _padding_frame, 'token_x': token_x, 'token_y': token_y, 'txt_msk': _padding_token_mask,
                'vid_msk_src': _padding_frame_mask, 'vid_msk_tgt': _padding_frame_mask,
                'cat_mask_x':  _padding_cat_mask, 'cat_mask_y': _padding_cat_mask
                }

    filenames = tf.io.gfile.glob(path)
    data, _ = split_files(filenames, slice_index, slice_count, params.data_seed * params.shuffle_input_filenames)
    decoder = decode_intstring if 'int64' in data[0] else decode_bytestring
    print('decode_intstring' if 'int64' in data[0] else 'decode_bytestring', data[0], len(data))

    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.repeat()

    data = data.interleave(lambda x: _text_decoder(decoder=decoder,
                                                   data=x,
                                                   ctx=time_patch_size * (language_token_per_frame - 1),
                                                   patch_size=language_token_per_frame - 1,
                                                   chunk_size=-1))

    data = data.shuffle(params.shuffle_buffer, seed=(params.data_seed if not params.use_random_dataloader else None))
    data = tf.data.Dataset.zip((data, padding_token, padding_frame, padding_frame_mask, padding_cat_mask))
    data = data.batch(sub_batch_size, num_parallel_calls=params.parallel_batch, deterministic=True)
    data = data.map(_memory_func, num_parallel_calls=tf2.data.AUTOTUNE)

    return data


def dataset_video(path: str, params: ModelParameter, sub_batch_size: int, slice_index, slice_count):
    """
    Creates a video dataset containing shuffled and prefetched windows.
    :param path: Path to dataset (in google cloud bucket)
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    three_axes = params.three_axes
    frame_height = params.frame_height
    frame_width = params.frame_width

    time_patch = params.time_patch
    color_channels = params.color_channels
    patch_size = params.patch_size
    n_ctx = params.n_ctx
    token_patch_size = params.token_patch_size
    language_token_patch = params.language_token_patch
    language_token_per_frame = params.language_token_per_frame

    time_patch_size = params.time_patch_size
    frame_height_patch = params.frame_height_patch
    frame_width_patch = params.frame_width_patch
    channel_color_size = params.channel_color_size

    assert not (language_token_per_frame > 0 and time_patch > 1), \
        ("Time patch and language token are currently not supported together")

    def _decode_func(name: tf.Tensor):
        data = tf.data.TFRecordDataset(filenames=tf.convert_to_tensor(name), buffer_size=2 ** 26, num_parallel_reads=1)
        data = data.map(frame_decoder, num_parallel_calls=1)

        data = data.window(size=n_ctx + time_patch, stride=1, shift=n_ctx, drop_remainder=True)
        data = data.interleave(interleave_func, cycle_length=1, num_parallel_calls=1, block_length=1)

        return data

    def _pre_func(*args):

        token_x, token_y, out_frame, frame_mask, \
        frame_mask_x, frame_mask_y, token_mask, token = (None, None, None, None, None, None, None, None)

        frame, concat, frame_mask, *args = args

        if params.use_language:
            token, token_mask, *args = args

        # frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, time_patch, frame_height_patch, patch_size,
        #                           frame_width_patch, patch_size, color_channels))

        # frame = tf.transpose(frame, [0, 1, 3, 5, 2, 4, 6, 7])

        if three_axes:
            out_frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                                           channel_color_size))
        else:
            out_frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, frame_height_patch * frame_width_patch,
                                           channel_color_size))

        concat = tf.reshape(concat, (sub_batch_size, time_patch_size + 1))
        concat = 1 - concat
        concat = tf.cast(concat, tf.bool)

        cat_mask_x = concat[:, :time_patch_size]
        cat_mask_y = concat[:, 1:time_patch_size + 1]

        frame_mask = tf.reshape(frame_mask, (sub_batch_size, time_patch_size + 1))
        frame_mask = 1 - frame_mask
        frame_mask = tf.cast(frame_mask, tf.bool)
        frame_mask_x = frame_mask[:, :time_patch_size]
        frame_mask_y = frame_mask[:, 1:time_patch_size + 1]

        if params.use_language:
            token = tf.reshape(token, (sub_batch_size, time_patch_size + 1, language_token_patch, token_patch_size))
            token = tf.cast(token, tf.int32)

            token_x = token[:, :time_patch_size]
            token_y = token[:, 1:time_patch_size + 1]

            token_mask = token_mask[:, 1:time_patch_size + 1]
            token_mask = tf.reshape(token_mask,
                                    (sub_batch_size, time_patch_size, language_token_patch, token_patch_size))
            token_mask = tf.cast(token_mask, tf.bool)

        return {k: v for k, v in {'frame':       out_frame, 'token_x': token_x, 'token_y': token_y,
                                  'vid_msk_src': frame_mask_x, 'vid_msk_tgt': frame_mask_y, 'txt_msk': token_mask,
                                  'cat_mask_x':  cat_mask_x, 'cat_mask_y': cat_mask_y
                                  }.items() if v is not None}

    if language_token_per_frame > 0:
        interleave_func = lambda x, y, z, a, b: tf.data.Dataset.zip((x, y, z, a, b)) \
            .batch(n_ctx + time_patch, drop_remainder=True,
                   num_parallel_calls=params.parallel_batch, deterministic=True)
    else:
        interleave_func = lambda x, y: tf.data.Dataset.zip((x, y)).batch(n_ctx + time_patch, drop_remainder=True,
                                                                         num_parallel_calls=params.parallel_batch,
                                                                         deterministic=True)

    frame_decoder = get_video_decoder(params,
                                      language_token_num_per_frame=language_token_per_frame,
                                      frame_height=frame_height, frame_width=frame_width,
                                      color_channels=color_channels,
                                      color_quantization_value=params.color_quantization_value)

    filenames = tf.io.gfile.glob(path)
    data: Dataset = tf.data.Dataset.from_tensor_slices(split_files(filenames, slice_index, slice_count,
                                                                   params.data_seed * params.shuffle_input_filenames)[
                                                           0])

    data = data.repeat()
    data = data.interleave(lambda x: _decode_func(x),
                           cycle_length=params.interleaved_datasets,
                           num_parallel_calls=tf2.data.AUTOTUNE)
    data = data.batch(sub_batch_size, num_parallel_calls=params.parallel_batch, deterministic=True)
    data = data.map(_pre_func, num_parallel_calls=tf2.data.AUTOTUNE)

    return data


def dataset(params: ModelParameter, sub_batch_size, slice_index, slice_count, _):
    """
    Creates any dataset containing shuffled and prefetched windows.
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    def memory_op(x):
        if not params.use_discrete_video_loss:
            x['frame'] = tf.cast(x['frame'], params.variable_dtype.activation_dtype) / \
                         (params.color_quantization_value - 1)
        else:
            x['frame'] = tf.cast(x['frame'], tf.int32)
        return x

    weights = []
    datasets = []

    for set in params.dataset_configs:
        dtype = set['type']
        path = set['path']
        weight = set['weight']

        if dtype != 'video' and dtype != 'text':
            raise ValueError(f"{dtype} is not a supported option for type for a dataset.")

        if dtype == 'video':
            datasets.append(dataset_video(path, params, sub_batch_size, slice_index, slice_count))
        elif dtype == 'text' and params.use_language:
            datasets.append(dataset_text(path, params, sub_batch_size, slice_index, slice_count))

        weights.append(weight)

    if len(datasets) > 1:
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        dset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    else:
        dset = datasets[0]

    if not params.use_bit_fold_input_pipeline:
        dset = dset.map(memory_op)
    dset = dset.map(align_tensor_op)

    return dset


def gpt_neo_input(params: ModelParameter, sub_batch_size: int, slice_index: int, slice_count: int, runs_log=None):
    params = ModelParameter(params)
    filenames = []
    for file in params.dataset_configs:
        filenames.extend(tf.io.gfile.glob(file['path']))

    filenames, skips = split_files(filenames, slice_index, slice_count,
                                   params.shuffle_input_filenames * params.data_seed, runs_log)

    dset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(filenames),
                                tf.data.Dataset.from_tensor_slices(skips)))

    if params.use_random_dataloader:
        dset = dset.repeat()

    def _memory_func(x):
        shp = (sub_batch_size, params.n_ctx // params.token_patch_size + params.output_offset, params.token_patch_size)
        x = tf.cast(tf.reshape(x, shp), tf.int32)
        if params.output_offset > 0:
            vals1 = x[:, :params.n_ctx]
            vals2 = x[:, params.output_offset:params.n_ctx + params.output_offset]
        else:
            vals1 = vals2 = x
        return {'token_x': vals1, 'token_y': vals2}

    decoder = decode_intstring if 'int64' in filenames[0] else decode_bytestring
    dset = dset.interleave(lambda x, _skip: _text_decoder(decoder, x, params.n_ctx,
                                                          params.token_patch_size * params.output_offset, -1,
                                                          params.shuffle_buffer * int(params.use_random_dataloader),
                                                          _skip),
                           cycle_length=params.interleaved_datasets,
                           num_parallel_calls=params.parallel_interleave)

    if params.use_random_dataloader:
        dset = dset.shuffle(params.shuffle_buffer,
                            seed=(params.data_seed if not params.use_random_dataloader else None))
    dset = dset.batch(sub_batch_size, num_parallel_calls=params.parallel_batch, deterministic=True)
    dset = dset.map(_memory_func)
    dset = dset.map(align_tensor_op)

    return dset
