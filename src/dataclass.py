"""
Contains a class as a datastore for model parameters
"""
import random
import typing

import mesh_tensorflow as mtf
import tensorflow as tf
from tensorflow.python.tpu.device_assignment import DeviceAssignment


class BlockConfig:
    def __init__(self, config, memory_reduction_strategy: str):
        if isinstance(config, BlockConfig):
            config = config.__dict__
        self.layer = []
        self.skip = False
        self.memory_reduction_strategy = memory_reduction_strategy
        self.__dict__.update(config)


class LearningRateConfig:
    def __init__(self, start_step: int = 0, final_step: int = 0, factor: float = 1.):
        self.start_step = start_step
        self.final_step = final_step
        self.factor = factor


class TensorStorage:
    def __init__(self):
        self.text_input_embedding: typing.Optional[mtf.Tensor] = None


class ModelParameter(typing.Dict[str, typing.Any]):
    def __init__(self, config: typing.Dict[str, typing.Any]):
        super().__init__()

        self.position_embedding = "absolute"  # "absolute" or "relative"(-learned) or "axial" | orthogonal for variables
        self.token_embedding = "absolute"
        self.empty_frame_embedding = "absolute"
        self.output_embedding = "absolute-orthogonal"  # embedding options above
        self.use_video = True
        self.save_graph = False
        self.use_language = True
        self.contrastive_across_samples = False
        self.contrastive_across_token_embeddings = False
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
        self.vocab_size = 256
        self.color_channels = 3
        self.three_axes = True
        self.dataset_configs = []
        self.data_seed = 456772
        self.parallel_batch = None
        self.parallel_interleave = None
        self.use_random_dataloader = False
        self.train = True
        self.debug_sample = False
        self.padding_token = 0
        self.concat_token = 4
        self.sequence_length = 32
        self.heads = 8
        self.features: typing.Optional[int] = None
        self.features_per_head: typing.Optional[int] = None
        self.depth = 16
        self.buffer_size = 4
        self.combine_assignments = False  # Needs more memory but it's faster
        self.shuffle_buffer = 256
        self.interleaved_datasets = 256
        self.token_patch_size = 1
        self.learning_rate = 5e-5
        self.storage_dtype = "float32"
        self.slice_dtype = "float32"
        self.calculation_dtype = "float32"
        self.optimizer_slice_dtype = "float32"
        self.optimizer_calculation_dtype = "float32"
        self.learning_rate_config = {}
        self.train_batch_size = 1
        self.grad_accumulation = 1
        self.macro_batching = 1
        self.macro_batch_loss_smoothing = False
        self.reduce_lr_on_plateau_timespan = 0
        self.reduce_lr_on_plateau_reduction = 2
        self.momentumnet_alpha = 0.99
        self.current_step = 0
        self.tpu_size = 32
        self.default_sleep_duration = 0.1
        self.lookahead_steps = 0
        self.lookahead_alpha = 0
        self.momentum = 0.95
        self.prefix = "datasets/full_hd_video"
        self.model_path = "gs://text-datasets/video-transformer/ctx=32-layer=64-heads=8-feat=256"
        self.tensorflow_optimization_settings = {"layout_optimizer": True,
                                                 "constant_folding": True,
                                                 "shape_optimization": True,
                                                 "remapping": True,
                                                 "arithmetic_optimization": True,
                                                 "dependency_optimization": True,
                                                 "loop_optimization": True,
                                                 "function_optimization": True,
                                                 "debug_stripper": True,
                                                 "scoped_allocator_optimization": True,
                                                 "pin_to_host_optimization": True,
                                                 "implementation_selector": True,
                                                 "auto_mixed_precision": True,
                                                 "disable_meta_optimizer": False,
                                                 "disable_model_pruning": False,
                                                 "min_graph_nodes": 0
                                                 }
        self.language_token_per_frame = 0
        self.weight_decay = 0.001
        self.vocab_weight_factorization = 0.125
        self.train_steps = 2 ** 30
        self.warmup_steps = 3000
        self.rezero_lr_multiplier = 0.1
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
        self.web_workers = 1
        self.equal_debugging_items_per_check = 16
        self.group_linear_factor = 2
        self.embedding_stddev = 0.04
        self.color_quantization_value = 256
        self.experts = 64
        self.pkm_axes = 2  # 2 axis = features^2 keys, 3 axis = features^3 keys...
        self.use_bit_fold_input_pipeline = False
        self.bit_fold_value = 4
        self.debug_train_step = False
        self.model_mode = 'jannet'
        self.optimizer = 'learning_rate'
        self.multi_loss_strategy = "linear"
        self.memory_reduction_strategy = "revnet"
        self.debug_gradients = False
        self.use_initial_position_embedding = False
        self.intermediate_feed_forward_multiplier = None
        self.intermediate_feed_forward_multiplier_multiplier = None
        self.own_color = "\x1b[32;1m"
        self.other_color = "\x1b[0m"
        self.scale_by_depth = True
        self.z_loss = 1e-4
        self.block_config = [{'layer': ["norm-group-shift-scale",
                                        "feed_forward-in_relu-group-in_glu_add-in_norm"]
                              },

                             {'layer': ["norm-group-std-shift-scale",
                                        "attention-in_relu-embedded-relative"]
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
        self.split_grad_accumulation = True
        self.log_dict_keys = []

        if hasattr(config, 'dict'):
            config = config.dict()

        for k, v in config.items():
            if k not in self.__dict__:
                print(f"WARNING: Unknown ModelParameter {k}={v!r}")
            self.__dict__[k] = v

        if self.grad_accumulation > 1:
            raise ValueError("Gradient accumulation is not supported right now. The optimizer was split into two "
                             "different 'sections' where the 'accumulation' section still has to be integrated")

        assert self.macro_batching > 0, "MacroBatching has to be >=1, where 1 means it's disabled"
        if isinstance(self.position_embedding, str):
            self.position_embedding = self.position_embedding.split('-')
            self.token_embedding = self.token_embedding.split('-')
            self.output_embedding = self.output_embedding.split('-')
            self.empty_frame_embedding = self.empty_frame_embedding.split('-')

            self.slice_dtype = getattr(tf, self.slice_dtype)
            self.storage_dtype = getattr(tf, self.storage_dtype)
            self.calculation_dtype = getattr(tf, self.calculation_dtype)
            self.optimizer_slice_dtype = getattr(tf, self.optimizer_slice_dtype)
            self.optimizer_calculation_dtype = getattr(tf, self.optimizer_calculation_dtype)

            self.learning_rate_config = {key: LearningRateConfig(**config) for key, config in
                                         self.learning_rate_config.items()}

            self.tensor_storage = TensorStorage()

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
        if self.features is None and self.features_per_head is None:
            raise ValueError("Either features or features_per_head has to be specified")
        if self.features is None:
            self.features = self.features_per_head * self.heads
        if self.features_per_head is None:
            self.features_per_head = self.features // self.heads
        if self.use_video and (self.frame_width * self.frame_height // self.patch_size) % self.experts:
            raise ValueError("Frame size has to be divisible by number of experts. Set \"experts\" to 1 if you're not "
                             "using MoE")
        if self.intermediate_feed_forward_multiplier_multiplier is not None:
            self.intermediate_feed_forward_multiplier = \
                self.group_linear_factor * self.intermediate_feed_forward_multiplier_multiplier / self.heads
        if self.intermediate_feed_forward_multiplier is None:
            self.intermediate_feed_forward_multiplier = self.group_linear_factor / self.heads
        if not self.use_video and self.language_token_per_frame != self.sequence_length:
            print(
                f"language_token_per_frame is unused in language-only mode. Overwriting with sequence_length={self.sequence_length}")
            self.language_token_per_frame = self.sequence_length
        if self.macro_batching > 1 and self.grad_accumulation > 1 and self.macro_batching % self.grad_accumulation != 0:
            raise ValueError(f'"macro_batching" needs do be divisible by "grad_accumulation", '
                             f'{self.macro_batching} is not divisible by {self.grad_accumulation}')

        if self.use_random_dataloader:
            print('WARNING: Use random dataset seed')
            for _ in range(random.randint(0, 1000)):
                self.data_seed = random.randint(0, 1000000)
        split_batch = self.heads < self.tpu_size
        split_heads = self.heads > 1
        self.mesh_shape = ','.join([f"b:{self.tpu_size // self.heads:.0f}"] * split_batch +
                                   [f"h:{self.heads:.0f}"] * split_heads)
        self.layout = ','.join([f"batch:b"] * split_batch +
                               [f"heads:h"] * split_heads)
        self.variable_dtype = mtf.VariableDType(self.storage_dtype, self.slice_dtype, self.calculation_dtype)
        self.optimizer_dtype = mtf.VariableDType(self.storage_dtype, self.optimizer_slice_dtype,
                                                 self.optimizer_calculation_dtype)
        self.block_config = [BlockConfig(conf, memory_reduction_strategy=self.memory_reduction_strategy) for conf in
                             self.block_config]
        self.input_block_config = [BlockConfig(conf, memory_reduction_strategy="checkpoint") for conf in
                                   self.input_block_config]
        self.output_block_config = [BlockConfig(conf, memory_reduction_strategy="checkpoint") for conf in
                                    self.output_block_config]
        self.time_patch_size = self.sequence_length // self.time_patch
        self.frame_height_patch = self.frame_height // self.patch_size
        self.frame_width_patch = self.frame_width // self.patch_size
        self.channel_color_size = self.color_channels * self.time_patch * self.patch_size ** 2
        self.fold_count = 32 // self.bit_fold_value
        if 2 ** self.bit_fold_value < self.color_quantization_value and self.use_bit_fold_input_pipeline:
            raise ValueError("when folding the input, the fold value must be qual or lager then the color bit value")
        self.language_token_patch = self.language_token_per_frame // self.token_patch_size
        if self.use_bit_fold_input_pipeline:
            self.channel_color_size = self.channel_color_size // self.fold_count

        self.product_key_value_vectors = self.features_per_head ** 2
        self.product_key_value_dim = mtf.Dimension("product_key_value_dim", self.product_key_value_vectors)
        self.head_dim = mtf.Dimension("heads", self.heads)
        self.head_dimensions = [self.head_dim]
        self.key_dim = mtf.Dimension("features_per_head", self.features // self.heads)
        self.sequence_per_head_dim = mtf.Dimension("sequence_per_head", self.time_patch_size // self.heads)
        self.pkm_dim = mtf.Dimension("pkm_axes", self.pkm_axes)

        self.feature_dims = self.head_dimensions + [self.key_dim]

        self.intermediate = [mtf.Dimension("intermediate",
                                           int(self.heads * self.key_dim.size *
                                               self.intermediate_feed_forward_multiplier))]
        self.expert_dim = mtf.Dimension("experts", self.experts)

        self.macro_batch_dim = mtf.Dimension("batch", self.train_batch_size * self.macro_batching)
        self.vocab_dim = mtf.Dimension('vocab', self.vocab_size)
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
        self.color_channel_dim = mtf.Dimension("color_channels", self.channel_color_size)
        frame_input_shape += [self.color_channel_dim]
        self.frame_input_shape = mtf.Shape(frame_input_shape)
        self.input_pipeline_shape = {}

        self.sequence_dim = mtf.Dimension("sequence", self.time_patch_size)
        self.token_patch_dim = mtf.Dimension("language_token_patch", self.token_patch_size)
        self.token_dim_shape = mtf.Shape([self.batch_dim,
                                          self.sequence_dim,
                                          self.token_patch_dim])
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
        self.variable_cache = {}
        self.cached_parameters = {}
        self.debug_outfeed = {}

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


class BlockArgs:
    def __init__(self, params: ModelParameter, tensor: typing.Optional[mtf.Tensor], name_extras: typing.List[str],
                 is_last: bool = False):
        self.params = params
        self.tensor = tensor
        self.name_extras = name_extras
        self.is_last = is_last

    def __call__(self, *args: typing.Union[ModelParameter, mtf.Tensor, typing.List[str], str]):
        new = BlockArgs(self.params, self.tensor, self.name_extras[:])
        for a in args:
            if isinstance(a, ModelParameter):
                new.params = a
            elif isinstance(a, mtf.Tensor):
                new.tensor = a
            elif isinstance(a, (list, tuple)):
                new.name_extras = list(a)
            elif isinstance(a, str):
                new.name_extras.append(str)
            else:
                raise ValueError(f"Argument {a} is of unsupported type {type(a)}. "
                                 f"Only ModelParameter, mtf.Tensor, typing.List[str] and str are supported")
        return new

    def __iter__(self):
        for itm in self.name_extras:
            yield itm

    def __len__(self):
        return len(self.name_extras)

    def __getitem__(self, idx: int):
        return self.name_extras[idx]
