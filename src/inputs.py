"""
Contains input pipeline code that generates tensorflow datasets if called
"""
import logging
import random
import re
from itertools import cycle
from functools import partial

import tensorflow.compat.v1 as tf
from tensorflow import contrib as tf_contrib
from tensorflow.data import Dataset

from .dataclass import ModelParameter, align_tensor_op


def split_files(path, slice_index, slice_count, seed):
    filenames = tf.io.gfile.glob(path)
    if not filenames:
        raise ValueError
    files = sorted(filenames)
    if seed != 0:
        random.seed(seed)
        random.shuffle(files)
    return files[slice_index::slice_count]


def get_video_decoder(language_token_num_per_frame=0, frame_height=None, frame_width=None, color_channels=None):
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
            'frame': tf.FixedLenFeature([], tf.string)
            }

    if decode_language_token:
        features.update({'tokens':     tf.FixedLenFeature([language_token_num_per_frame], tf.int64),
                         'skip_frame': tf.FixedLenFeature([], tf.int64),
                         'mask':       tf.FixedLenFeature([], tf.int64)
                         })

    def frame_decoder(proto):
        '''
        :param proto: Proto buffer to be decoded.
        :return: tensor with decode frame.

        This Function will decode frame from proto buffer.
        '''

        sample = tf.parse_single_example(proto, features)
        frame = tf.image.decode_image(sample['frame'])

        if decode_language_token:
            tokens = sample['tokens']
            skip_frame = sample['skip_frame']
            mask = sample['skip_frame']

            if skip_frame > 0:
                frame = tf.zeros(shape=(frame_height, frame_width, color_channels), dtype=tf.uint8)

            b_mask = tf.less_equal(token_range, tf.cast(mask, tf.int32))

            return frame, tokens, skip_frame, b_mask

        return frame

    return tf.function(frame_decoder, experimental_compile=False)


def _read_records(decoder, filenames: tf.Tensor):
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    return tf.data.TFRecordDataset(filenames=filenames, buffer_size=buffer_size)


def _text_decoder(decoder, data: tf.Tensor, ctx: int, patch_size: int, chunk_size: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param data: protobuf object to decode
    :param ctx: context size of generated dataset
    :param chunk_size: batch size directly after creating the dataset
    :return: tensorflow dataset of token
    """

    def chunk(tfrecorddataset):
        data = decoder(tfrecorddataset)
        if chunk_size > 0:
            data = data.batch(chunk_size)
        data = data.window(size=ctx + patch_size, shift=ctx, stride=1, drop_remainder=True)
        data = data.interleave(lambda x: x.batch(ctx + patch_size, drop_remainder=True))
        return data

    return tf.data.TFRecordDataset(filenames=data).interleave(chunk)



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

    padding_token_mask = tf.ones((time_patch_size, language_token_patch, token_patch_size), dtype=tf.bool)
    padding_token_mask = tf.data.Dataset.from_tensors(padding_token_mask).repeat()

    def _memory_func(x, _padding_token, _padding_frame, _padding_frame_mask, _padding_token_mask):

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

        _padding_token_mask = tf.reshape(_padding_token_mask,
                                         (sub_batch_size, time_patch_size, language_token_patch, token_patch_size))

        return {'frame':   _padding_frame, 'token_x': token_x, 'token_y': token_y, 'txt_msk': _padding_token_mask,
                'vid_msk_src': _padding_frame_mask, 'vid_msk_tag': _padding_frame_mask,
                }

    data = split_files(path, slice_index, slice_count, params.data_seed * params.shuffle_input_filenames)
    decoder = decode_intstring if 'int64' in data[0] else decode_bytestring
    print('decode_intstring' if 'int64' in data[0] else 'decode_bytestring', data[0], len(data))

    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.repeat()

    data = data.interleave(lambda x: _text_decoder(decoder=decoder,
                                                   data=x,
                                                   ctx=time_patch_size * (language_token_per_frame - 1),
                                                   patch_size=(language_token_per_frame - 1),
                                                   chunk_size=-1))

    data = data.shuffle(params.shuffle_buffer, seed=params.data_seed)
    data = tf.data.Dataset.zip((data, padding_token, padding_frame, padding_frame_mask, padding_token_mask))
    data = data.batch(sub_batch_size)
    data = data.map(_memory_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

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

        token_x, token_y, out_frame, frame_mask,\
        frame_mask_x, frame_mask_y, token_mask, token = (None, None, None, None, None, None, None, None)

        frame, *args = args

        if params.use_language:
            token, frame_mask, token_mask, *args = args

        frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, time_patch, frame_height_patch, patch_size,
                                   frame_width_patch, patch_size, color_channels))

        frame = tf.transpose(frame, [0, 1, 3, 5, 2, 4, 6, 7])

        if three_axes:
            out_frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, frame_height_patch, frame_width_patch,
                                           channel_color_size))
        else:
            out_frame = tf.reshape(frame, (sub_batch_size, time_patch_size + 1, frame_height_patch * frame_width_patch,
                                           channel_color_size))

        if params.use_language:
            token = tf.reshape(token, (sub_batch_size, time_patch_size + 1, language_token_patch, token_patch_size))
            token = tf.cast(token, tf.int32)

            token_x = token[:, :time_patch_size]
            token_y = token[:, 1:time_patch_size + 1]

            frame_mask = tf.reshape(frame_mask, (sub_batch_size, time_patch_size + 1))
            frame_mask = 1 - frame_mask
            frame_mask = tf.cast(frame_mask, tf.bool)
            frame_mask_x = frame_mask[:, :time_patch_size]
            frame_mask_y = frame_mask[:, 1:time_patch_size + 1]

            token_mask = token_mask[:, 1:time_patch_size + 1]
            token_mask = tf.reshape(token_mask,
                                    (sub_batch_size, time_patch_size, language_token_patch, token_patch_size))
            token_mask = tf.cast(token_mask, tf.bool)

        return {k: v for k, v in {'frame':   out_frame, 'token_x': token_x, 'token_y': token_y,
                                  'vid_msk_src': frame_mask_x, 'vid_msk_tag': frame_mask_y,
                                  'txt_msk': token_mask}.items() if v is not None}

    if language_token_per_frame > 0:
        interleave_func = lambda x, y, z, a: tf.data.Dataset.zip((x, y, z, a)) \
            .batch(n_ctx + time_patch, drop_remainder=True)
    else:
        interleave_func = lambda x: x.batch(n_ctx + time_patch, drop_remainder=True)

    frame_decoder = get_video_decoder(language_token_num_per_frame=language_token_per_frame,
                                      frame_height=frame_height, frame_width=frame_width, color_channels=color_channels)

    data: Dataset = tf.data.Dataset.from_tensor_slices(split_files(path, slice_index, slice_count,
                                                                   params.data_seed * params.shuffle_input_filenames))

    data = data.repeat()
    data = data.interleave(lambda x: _decode_func(x),
                           cycle_length=params.interleaved_datasets,
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(sub_batch_size)
    data = data.map(_pre_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return data


def dataset(params: ModelParameter, sub_batch_size, slice_index, slice_count):
    """
    Creates any dataset containing shuffled and prefetched windows.
    :param params: ModelParameter
    :return: tensorflow dataset
    """

    def memory_op(x):
        x['frame'] = tf.cast(x['frame'], params.calculation_dtype) / 255
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
        elif dtype == 'text':
            datasets.append(dataset_text(path, params, sub_batch_size, slice_index, slice_count))

        weights.append(weight)

    if len(datasets) > 1:
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        dset = tf.data.experimental.sample_from_datasets(datasets, weights=weights, seed=params.data_seed)
    else:
        dset = datasets[0]

    dset = dset.map(memory_op)
    dset = dset.map(align_tensor_op)
    dset = dset.skip(params.current_step)

    return dset


def _get_number_of_documents(filename):
    # extracts number of files from a filename formatted "<name>_<num_documents>.tfrecords."
    # if no pattern is matched, returns None
    match = re.search("_(\d{1,}).tfrecords$", filename)
    return int(match.group(1)) if match is not None else match


def _get_number_of_documents_by_iteration(filename):
    # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
    # this could be very slow.
    logging.warning(
            "inputs/sequential_input() found no metadata found in filename - iterating through first tfrecord to find global length")
    count = 0
    for item in tf.io.tf_record_iterator(filename):
        count += 1
    return count


def _get_skip_index(all_files, n_batches):
    prev_cumsum = 0
    cumsum = 0
    global_n_documents = None
    for count, f in cycle(enumerate(all_files)):
        prev_cumsum = cumsum
        if _get_number_of_documents(f) is not None:
            cumsum += _get_number_of_documents(f)
        elif global_n_documents is None:
            global_n_documents = _get_number_of_documents_by_iteration(f)
            cumsum += global_n_documents
        else:
            cumsum += global_n_documents
        if cumsum == n_batches:
            remainder = 0
            skip_idx = count + 1
        elif cumsum > n_batches:
            remainder = n_batches - prev_cumsum
            skip_idx = count
            break
    return skip_idx, remainder


def gpt_neo_input_old(params, sub_batch_size, slice_index, slice_count):
    """
    Input fn that reads tfrecords encoded with a fixed chunk size (== n_ctx + 1), and that either:

        - has the number of documents for each tfrecord file encoded in the title in the format
          <name>_<n_documents>.tfrecords.

          OR

        - has a fixed number of documents per tfrecord file.

    If the glob pattern above isn't matched, we assume that each document has the same number of samples as the first
    tfrecord read.
    If this isn't the case, it may result in errors, or some samples being missed.

    This means we can calculate the number of samples we've seen so far using the global step,
    and can use dataset.skip() to iterate through the list of filenames, as opposed to the whole dataset, which is
    incredibly inefficient.

    If training is starting and stopping often, as with TPU pre-emption, reading the whole dataset sequentially appears
    to improve model
    performance, as it results in less repeated data.
    :param params: serialized dict of ModelParameter instance
    :return: tensorflow dataset
    """

    params = ModelParameter(params)

    filenames = []
    for config in params.dataset_configs:
        filenames.extend(split_files(config['path'], slice_index, slice_count,
                                     params.shuffle_input_filenames * params.data_seed))

    # repeat filenames to infinity
    dset: Dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()

    def _memory_func(x):
        x = tf.reshape(x, (sub_batch_size, params.n_ctx // params.token_patch_size + 1, params.token_patch_size))
        x = tf.cast(x, tf.int32)

        vals1 = x[:, :params.n_ctx]
        vals2 = x[:, 1:params.n_ctx + 1]

        return {'token_x': vals1, 'token_y': vals2}

    decoder = decode_intstring if 'int64' in filenames[0] else decode_bytestring
    dset = dset.interleave(lambda x: _text_decoder(decoder, x, params.n_ctx, params.token_patch_size, -1))

    dset = dset.shuffle(params.shuffle_buffer, seed=params.data_seed)
    dset = dset.batch(sub_batch_size)

    dset = dset.map(_memory_func)
    dset = dset.map(align_tensor_op)

    return dset



def _text_decoder_new(decoder, data: tf.Tensor, ctx: int, patch_size: int, chunk_size: int):
    """
    Read a given tfrecord and windowed text dataset out of it.
    :param data: protobuf object to decode
    :param ctx: context size of generated dataset
    :param chunk_size: batch size directly after creating the dataset
    :return: tensorflow dataset of token
    """

    def chunk(tfrecorddataset):
        data = decoder(tfrecorddataset)
        if chunk_size > 0:
            data = data.batch(chunk_size)
        data = data.window(size=ctx + patch_size, shift=ctx, stride=1, drop_remainder=True)
        data = data.interleave(lambda x: x.batch(ctx + patch_size, drop_remainder=True))
        return data

    return data.interleave(chunk)

@tf.function
def decode_bytestring_static(proto):
    text_slice = tf.parse_single_example(proto, {'text': tf.FixedLenFeature([], tf.string)})['text']
    return tf.strings.unicode_decode(text_slice, 'UTF-8')


@tf.function
def decode_intstring_static(proto):
    x = tf.parse_single_example(proto, {'text': tf.VarLenFeature(tf.int64)})
    x = x['text']
    x = tf.sparse.to_dense(x)
    x = tf.cast(x, tf.int32)
    return x

def ds_set_shape(ds, shape):
  def inner(x):
    x.set_shape(shape)
    return x
  return ds.map(inner)


def ds_chunkify(ds, chunk_size):
  ds = ds.unbatch()
  ds = ds.batch(chunk_size)
  ds = ds_set_shape(ds, [chunk_size])
  return ds


# def randi(maxval, shape=[], dtype=tf.dtypes.int32):
#   return tf.random.uniform(shape, maxval=maxval, dtype=dtype)

# def verify(op_name, *args, **kws):
#   fn = getattr(tf.debugging, 'assert_' + op_name)
#   op = fn(*args, **kws)
#   return tf.control_dependencies([op])

# verify_less_eq = partial(verify, "less_equal")

# def randrange(haystack_size, needle_size, batch_size=None, dtype=tf.dtypes.int32):
#   with verify_less_eq(needle_size, haystack_size):
#     if batch_size is not None:
#       shape = [batch_size]
#     else:
#       shape = []
#     return tf.range(randi(haystack_size - needle_size, shape=shape, dtype=dtype))



def verify(op_name, *args, **kws):
  fn = getattr(tf.debugging, 'assert_' + op_name)
  op = fn(*args, **kws)
  return tf.control_dependencies([op])

verify_less_eq = partial(verify, "less_equal")


def shapeof(shape=None):
  if hasattr(shape, 'shape'):
    shape = shape.shape
  if isinstance(shape, tf.TensorShape):
    shape = shape.as_list()
  if shape is None:
    shape = [1]
  if isinstance(shape, int):
    shape = [shape]
  assert isinstance(shape, (list, tuple))
  return shape

# given a bin that holds `total` elements, return a random
# position such that you can take the next `subset` elements
# without going out of bounds. E.g. randpos(1,10) will return
# [0..9], randpos(2,10) will return [0..8], etc.
def randpos(subset, total, shape=[1], *, dtype=tf.int64):
  shape = shapeof(shape)
  #assert subset <= total
  with verify_less_eq(subset, total):
    return tf.random.uniform(shape=shape, maxval=(total - subset)+1, dtype=dtype)

def randrange(subset, total, shape=[1], *, dtype=tf.int64):
  subset = tf.saturate_cast(subset, dtype)
  total = tf.saturate_cast(total, dtype)
  shape = shapeof(shape)
  pos = randpos(subset, total, shape=shape, dtype=dtype)
  part = tf.tile(tf.expand_dims(tf.range(subset, dtype=dtype), axis=0), shape + [1])
  indices = part + tf.expand_dims(pos, axis=1)
  return indices

def sample_from(x, amount, shape=[1]):
  x = tf.convert_to_tensor(x)
  if x.dtype == tf.string:
    x = tf.strings.unicode_decode(x, 'utf-8')
    x = tf.cast(x, tf.int32)
  shape = shapeof(shape)
  n = tf.size(x)
  indices = randrange(amount, n, shape=shape)
  return tf.gather(x, indices)
  

import os
homepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def random_text():
  with open(os.path.join(homepath, 'LICENSE')) as f:
    return f.read()

def sample_random_text(n_ctx, batch_size, text=None):
  if text is None:
    text = random_text()
  assert len(text) >= n_ctx
  chars = tf.strings.unicode_decode(text, 'utf-8')
  tokens = tf.saturate_cast(chars, tf.int32)
  return sample_from(tokens, n_ctx, batch_size)


# Sample 1024(+1) tokens from the stitched together text
def sample_text(x, amount, maxlen=None):
    if maxlen is None:
      maxlen = tf.size(x)
    with verify_less_eq(amount, maxlen):
      r = randi(maxlen-amount)
    r = tf.range(r, r+amount)
    r = tf.reshape(r, [amount])
    return tf.gather(x, r)

from functools import partial

def gpt_random_input(params, sub_batch_size, slice_index, slice_count):
    ds = tf.data.Dataset.from_tensor_slices([random_text()])
    ds = ds.repeat()
    def sampler(x):
      x = sample_from(x, params.n_ctx + 1)
      x = tf.squeeze(x, 0)
      x = tf.expand_dims(x, -1)
      return x[0:-1], x[1:]
    ds = ds.map(sampler, num_parallel_calls=params.num_parallel_calls)
    ds = ds.batch(sub_batch_size)
    def set_shape(*args):
      for x in args:
        shape = shapeof(x)
        assert shape[0] is None or shape[0] == sub_batch_size
        shape[0] = sub_batch_size
        x.set_shape(shape)
      return args
    ds = ds.map(set_shape, num_parallel_calls=params.num_parallel_calls)
    return ds

def gpt_neo_input(params, sub_batch_size, slice_index, slice_count):
    """
    Input fn that reads tfrecords encoded with a fixed chunk size (== n_ctx + 1), and that either:

        - has the number of documents for each tfrecord file encoded in the title in the format
          <name>_<n_documents>.tfrecords.

          OR

        - has a fixed number of documents per tfrecord file.

    If the glob pattern above isn't matched, we assume that each document has the same number of samples as the first
    tfrecord read.
    If this isn't the case, it may result in errors, or some samples being missed.

    This means we can calculate the number of samples we've seen so far using the global step,
    and can use dataset.skip() to iterate through the list of filenames, as opposed to the whole dataset, which is
    incredibly inefficient.

    If training is starting and stopping often, as with TPU pre-emption, reading the whole dataset sequentially appears
    to improve model
    performance, as it results in less repeated data.
    :param params: serialized dict of ModelParameter instance
    :return: tensorflow dataset
    """

    params = ModelParameter(params)
    self = params

    filenames = []
    for config in params.dataset_configs:
        filenames.extend(split_files(config['path'], slice_index, 1024,
                                     params.shuffle_input_filenames * params.data_seed))

    # repeat filenames to infinity
    filenames_ds: Dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if self.is_training and not self.cache:
      filenames_ds = filenames_ds.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    tfrecorddataset = filenames_ds.apply(
        tf_contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_readers, sloppy=True))
    dataset = tfrecorddataset

    # def chunk(data):
    #     if chunk_size > 0:
    #         data = data.batch(chunk_size)
    #     data = data.window(size=ctx + patch_size, shift=ctx, stride=1, drop_remainder=True)
    #     data = data.interleave(lambda x: x.batch(ctx + patch_size, drop_remainder=True))
    #     return data
    chunk = partial(ds_chunkify, chunk_size=params.chunk_size)
    sample = partial(sample_text, amount=params.n_ctx + params.token_patch_size, maxlen=params.chunk_size)

    dataset_parser_static = decode_intstring_static if 'int64' in filenames[0] else decode_bytestring_static

    def _memory_func(x):
        # x = tf.reshape(x, (params.n_ctx // params.token_patch_size + 1, params.token_patch_size))
        # x = tf.cast(x, tf.int32)
        vals1 = x[..., :-1]
        vals2 = x[..., 1:]
        # TODO: Why is tf.newaxis necessary here?
        vals1 = vals1[..., tf.newaxis]
        vals2 = vals2[..., tf.newaxis]
        return {'token_x': vals1, 'token_y': vals2}

    def dataset_parser_dynamic(x):
      x = sample(x)
      x = _memory_func(x)
      x = align_tensor_op(x)
      return x

    def dataset_parser(x):
      x = dataset_parser_static(x)
      raise NotImplementedError("TODO: windowing")
      x = dataset_parser_dynamic(x)
      return x

    # tfrecorddataset.map(dataset_parser_static).unbatch().window(size=params.n_ctx + params.token_patch_size, shift=params.n_ctx, stride=1, drop_remainder=True).batch(params.n_ctx + params.token_patch_size, drop_remainder=True)

    if self.is_training and self.cache_decoded:
      dataset = dataset.map(
          dataset_parser_static,
          num_parallel_calls=self.num_parallel_calls)
      dataset = dataset.apply(chunk)
        
    if self.cache:
      dataset = dataset.cache()
    
    if self.is_training:
      # We shuffle only during training, and during training, we must produce an
      # infinite dataset, so apply the fused shuffle_and_repeat optimized
      # dataset transformation.
      dataset = dataset.apply(
          tf_contrib.data.shuffle_and_repeat(params.shuffle_buffer)) #(1024 * 16))

    if not self.is_training:
      # Padding for eval.
      dataset = self.pad_dataset(dataset, num_hosts)


    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    if self.is_training and self.cache_decoded:
      dataset = dataset.apply(
          tf_contrib.data.map_and_batch(
              dataset_parser_dynamic,
              batch_size=sub_batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))
    else:
      dataset = dataset.apply(
          tf_contrib.data.map_and_batch(
              dataset_parser,
              batch_size=sub_batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))

    # Assign static batch size dimension
    #dataset = dataset.map(partial(set_shapes, sub_batch_size))
    #import pdb; pdb.set_trace()

    # Prefetch overlaps in-feed with training
    if self.prefetch_depth_auto_tune:
      dataset = dataset.prefetch(tf_contrib.data.AUTOTUNE)
    else:
      dataset = dataset.prefetch(4)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    return dataset

    

def foo():
    def _memory_func(x):
        x = tf.reshape(x, (sub_batch_size, params.n_ctx // params.token_patch_size + 1, params.token_patch_size))
        x = tf.cast(x, tf.int32)

        vals1 = x[:, :params.n_ctx]
        vals2 = x[:, 1:params.n_ctx + 1]

        return {'token_x': vals1, 'token_y': vals2}

    dset = dset.interleave(lambda x: _text_decoder(decoder, x, params.n_ctx, params.token_patch_size, -1))

    # dset = dset.shuffle(params.shuffle_buffer, seed=params.data_seed)
    dset = dset.batch(sub_batch_size)

    dset = dset.map(_memory_func)
    dset = dset.map(align_tensor_op)

    return dset
