"""
Contains a class as a datastore for model parameters
"""
import typing
import random

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu.device_assignment import DeviceAssignment


class BlockConfig:
    def __init__(self, config, use_revnet=True):
        if isinstance(config, BlockConfig):
            config = config.__dict__
        self.layer = []
        self.skip = False
        self.use_revnet = use_revnet
        self.__dict__.update(config)


class ModelParameter(typing.Dict[str, typing.Any]):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__()

        self.use_video = True
        self.save_graph = False
        self.use_language = True
        self.input_dropout = 0.
        self.output_offset = 1
        self.weight_standardisation = True
        self.use_checkpointing = False
        self.max_checkpoints_keep = 1
        self.steps_per_checkpoint = 100_000
        self.time_patch = 1
        self.patch_size = 16
        self.frame_width = 320
        self.frame_height = 176
        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_epsilon = 1e-6
        self.adaptive_gradient_clipping = True
        self.vocab_size = 256
        self.color_channels = 3
        self.three_axes = True
        self.dataset_configs = []
        self.data_seed = 456772
        self.train = True
        self.debug_sample = False
        self.padding_token = 0
        self.concat_token = 4
        self.n_ctx = 32
        self.n_head = 8
        self.n_embd: typing.Optional[int] = None
        self.n_embd_per_head: typing.Optional[int] = None
        self.n_blocks = 16
        self.buffer_size = 4
        self.shuffle_buffer = 256
        self.interleaved_datasets = 256
        self.token_patch_size = 1
        self.learning_rate = 5e-5
        self.storage_dtype = "float32"
        self.slice_dtype = "float32"
        self.calculation_dtype = "float32"
        self.train_batch_size = 1
        self.grad_accumulation = 1
        self.macro_batching = 1
        self.current_step = 0
        self.batch_splits = 1
        self.head_splits = 32.
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.tensorflow_optimization_settings = {"layout_optimizer":              True,
                                                 "constant_folding":              True,
                                                 "shape_optimization":            True,
                                                 "remapping":                     True,
                                                 "arithmetic_optimization":       True,
                                                 "dependency_optimization":       True,
                                                 "loop_optimization":             True,
                                                 "function_optimization":         True,
                                                 "debug_stripper":                True,
                                                 "disable_model_pruning":         False,
                                                 "scoped_allocator_optimization": True,
                                                 "pin_to_host_optimization":      True,
                                                 "implementation_selector":       True,
                                                 "auto_mixed_precision":          True,
                                                 "disable_meta_optimizer":        False,
                                                 "min_graph_nodes":               0
                                                 }
        self.language_token_per_frame = 0
        self.weight_decay = 0.001
        self.train_steps = 150_000
        self.warmup_steps = 3000
        self.learning_rate_decay_multi = 1
        self.convolution_size = 16
        self.learning_rate_decay_start_step = 100_000
        self.learning_rate_decay_min = 5e-10
        self.iterations = 2500
        self.initial_autoregressive_position = 128
        self.use_autoregressive_sampling = False
        self.sampling_temperature = 0
        self.weight_centralisation = True
        self.shuffle_input_filenames = True
        self.calc_accuracy = False
        self.num_of_sample = 10
        self.gradient_clip = -1
        self.group_linear_factor = 2
        self.embedding_stddev = 0.04
        self.color_quantization_value = 256
        self.use_discrete_video_loss = False
        self.debug_train_step = False
        self.model_mode = 'jannet'
        self.optimizer = 'adam'
        self.multi_loss_strategy = "linear"
        self.use_revnet = True
        self.debug_gradients = False
        self.use_initial_position_embedding = False
        self.intermediate_feed_forward_multiplier = None
        self.own_color = "\x1b[32;1m"
        self.other_color = "\x1b[0m"
        self.block_config = [{'layer': ["norm-group-instance-mean-std-shift-scale",
                                        "feed_forward-relu-group"]
                              },

                             {'layer': ["norm-group-instance-mean-std-shift-scale",
                                        "attention-relu-embedded-kernel_softmax"]
                              }]

        self.input_block_config = []
        self.output_block_config = []

        self.mesh: typing.Optional[mtf.Mesh] = None
        self.d_assignment: typing.Optional[DeviceAssignment] = None
        self.mesh_impl: typing.Optional[mtf.simd_mesh_impl.SimdMeshImpl] = None
        self.num_cores = 0
        self.num_hosts = 0
        self.num_cores_per_host = 0
        self.masked_attention_dimensions = [0]
        self.log_dict_keys = []

        if hasattr(config, 'dict'):
            config = config.dict()

        for k, v in config.items():
            if k not in self.__dict__:
                print(f"WARNING: Unknown ModelParameter {k}={v!r}")
            self.__dict__[k] = v

        self.multi_loss_strategy = self.multi_loss_strategy.lower()
        if self.multi_loss_strategy not in ["linear", "pcgrad", "mgda"]:
            print(f'{self.multi_loss_strategy} is not in the support option list for multi loss strategies: '
                  f'["linear", "pcgrad", "mgda"]. default to "linear".')
            self.multi_loss_strategy = "linear"
        if not self.use_language and not self.use_video:
            raise ValueError("Language and video mode are disabled. No model can be built.")
        if self.weight_standardisation and not self.weight_centralisation:
            print("Can't standardise weights without centralizing them first. Enabling it.")
            self.weight_centralisation = True
        if self.n_embd is None and self.n_embd_per_head is None:
            raise ValueError("Either n_embd or n_embd_per_head has to be specified")
        if self.n_embd is None:
            self.n_embd = self.n_embd_per_head * self.n_head
        if self.n_embd_per_head is None:
            self.n_embd_per_head = self.n_embd // self.n_head
        if isinstance(self.storage_dtype, str):
            self.storage_dtype = getattr(tf, self.storage_dtype)
        if isinstance(self.slice_dtype, str):
            self.slice_dtype = getattr(tf, self.slice_dtype)
        if isinstance(self.calculation_dtype, str):
            self.calculation_dtype = getattr(tf, self.calculation_dtype)
        if self.intermediate_feed_forward_multiplier is None:
            self.intermediate_feed_forward_multiplier = self.group_linear_factor / self.head_splits
        if not self.use_video and self.language_token_per_frame != self.n_ctx:
            print(f"language_token_per_frame is unused in language-only mode. Overwriting with n_ctx={self.n_ctx}")
            self.language_token_per_frame = self.n_ctx
        if self.macro_batching > 1 and self.grad_accumulation > 1 and self.macro_batching % self.grad_accumulation != 0:
            raise ValueError(f'"macro_batching" needs do be divisible by "grad_accumulation", '
                             f'{self.macro_batching} is not divisible by {self.grad_accumulation}')

        self.inf_data = False
        if self.data_seed == 0:
            print('WARNING: Use random dataset seed')
            self.inf_data = True
            for _ in range(random.randint(0, 1000)):
                self.data_seed = random.randint(0, 1000000)

        split_batch = self.batch_splits > 1
        split_heads = self.head_splits > 1
        if not hasattr(self, 'split_vocab'):
            self.split_vocab = split_heads and isinstance(self.head_splits, int) and self.vocab_size > 256
            if self.split_vocab:
                full_partition_size = self.head_splits * 128
                self.vocab_size += full_partition_size - self.vocab_size % full_partition_size
                self.vocab_size //= self.n_head
            elif self.vocab_size % 256 > 0:
                self.vocab_size += 256 - self.vocab_size % 256
        self.mesh_shape = ','.join([f"b:{self.batch_splits:.0f}"] * split_batch +
                                   [f"h:{self.head_splits:.0f}"] * split_heads)
        self.layout = ','.join([f"batch:b"] * split_batch +
                               [f"heads:h"] * split_heads)
        self.variable_dtype = mtf.VariableDType(self.storage_dtype, self.slice_dtype, self.calculation_dtype)
        self.block_config = [BlockConfig(conf, use_revnet=self.use_revnet) for conf in self.block_config]
        self.input_block_config = [BlockConfig(conf, use_revnet=False) for conf in self.input_block_config]
        self.output_block_config = [BlockConfig(conf, use_revnet=False) for conf in self.output_block_config]
        self.time_patch_size = self.n_ctx // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.language_token_patch = self.language_token_per_frame // self.token_patch_size

        self.head_dim = mtf.Dimension("heads", self.n_head)
        self.head_dimensions = [self.head_dim]
        self.key_dim = mtf.Dimension("features_per_head", self.n_embd // self.n_head)
        self.sequence_per_head_dim = mtf.Dimension("sequence_per_head", self.time_patch_size // self.n_head)

        self.feature_dims = self.head_dimensions + [self.key_dim]

        self.intermediate = [mtf.Dimension("intermediate",
                                           int(self.n_head * self.key_dim.size *
                                               self.intermediate_feed_forward_multiplier))]

        self.vocab_dim = mtf.Dimension("vocab", self.vocab_size)
        self.vocab_dims = [self.head_dim] * self.split_vocab + [self.vocab_dim]
        self.batch_dim = mtf.Dimension("batch", self.train_batch_size)
        self.frame_input_sequence = mtf.Dimension("_sequence", self.time_patch_size + 1)

        frame_input_shape = [self.batch_dim, self.frame_input_sequence]

        if self.three_axes:
            frame_input_shape += [
                    mtf.Dimension("height", self.frame_height_patch),
                    mtf.Dimension("width", self.frame_width_patch),
                    ]

        else:
            frame_input_shape += [
                    mtf.Dimension(
                            "height", self.frame_height_patch * self.frame_width_patch
                            )
                    ]

        frame_input_shape += [mtf.Dimension("color_channels", self.channel_color_size)]
        self.frame_input_shape = mtf.Shape(frame_input_shape)
        self.input_pipeline_shape = {}

        self.sequence_dim = mtf.Dimension("sequence", self.time_patch_size)
        self.token_dim_shape = mtf.Shape([self.batch_dim,
                                          self.sequence_dim,
                                          mtf.Dimension("language_token_patch", self.token_patch_size)])
        self.frame_mask_shape = mtf.Shape([self.batch_dim, self.sequence_dim])

        if self.use_video:
            self.input_pipeline_shape['frame'] = self.frame_input_shape
            self.input_pipeline_shape['cat_mask_x'] = self.frame_mask_shape
            self.input_pipeline_shape['cat_mask_y'] = self.frame_mask_shape
            self.input_pipeline_shape['vid_msk_src'] = self.frame_mask_shape
            self.input_pipeline_shape['vid_msk_tgt'] = self.frame_mask_shape

            self.discrete_dim = [mtf.Dimension("discrete", self.channel_color_size * self.color_quantization_value)]
            self.discrete_color_dim = mtf.Dimension("color_quantization", self.color_quantization_value)

        if self.use_language:
            self.input_pipeline_shape['token_x'] = self.token_dim_shape
            self.input_pipeline_shape['token_y'] = self.token_dim_shape

        if self.use_language and self.use_video:
            self.token_dim_shape._dims.insert(2, mtf.Dimension("height", self.language_token_patch))
            self.input_pipeline_shape['txt_msk'] = self.token_dim_shape

        self.input_pipeline_shape = align_tensor_op(self.input_pipeline_shape)
        self.attention_idx = 0

    def __getitem__(self, key: str) -> typing.Any:
        print(f"Getting {key} via deprecated interface")
        return self.key

    def __setitem__(self, key: str, value: typing.Any) -> None:
        print(f"Setting {key} via deprecated interface")
        self.key = value

    def get(self, key: str, default: typing.Any) -> typing.Any:
        """
        Default python get from list
        :param key: key to check for in dictionary
        :param default: default value if key doesn't exist
        :return: whatever value belongs to the key or the default
        """
        print(f"Getting {key} via deprecated interface with default value {default}")
        return self.__dict__.get(key, default)

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

    def dict(self) -> typing.Dict[str, typing.Any]:
        """
        :return: dictionary containing parameters
        """
        return self.__dict__


def align_tensor_op(x):
    tensors = []
    if 'frame' in x:
        tensors.extend([x['frame'], x['cat_mask_x'], x['cat_mask_y']])
        tensors.extend([x['vid_msk_src'], x['vid_msk_tgt']])
    if 'token_x' in x:
        tensors.extend([x['token_x'], x['token_y']])
    if 'txt_msk' in x:
        tensors.append(x['txt_msk'])
    return tensors
