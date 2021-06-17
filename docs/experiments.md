# Transformer Experiments

Relevant: [Do Transformer Modifications Transfer Across Implementations and Applications?](https://arxiv.org/abs/2102.11972)

## Long List

| Category | Title | Description | Loss | Time to converge | Eval steps/s | Baseline | Recommended | Tested | References |
| --- | ---| ---| ---| ---| ---| ---| --- | --- | --- |
| Routing | Attention | Default routing mechanism used by transformers | ? | ? | ? | ✔️️ | ✔️️ |  ❌️ | [Attention is all you need](https://arxiv.org/abs/1706.03762) |
| Routing | Local Attention | (dilated) local attention windows | ? | ? | ? | ✔️️ | ✔️️ |  ❌️ | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) |
| Routing | Spatial MLP | Spatial multi-layer perception | ? | ? | ? | ✔️️ | ✔️️ |  ❌️ | mixer |
| Routing | MLP | Generate spatial weights using local features | ? | ? | ? | ✔️️ | ✔️️ |  ❌️ | dense synthesizer |
| Routing | Convolution | Additional routing used by Conformer | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) |
| Routing | Lightweight Convolution | Convolution with adaptively generated weights | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Pay Less Attention with Lightweight and Dynamic Convolutions](https://arxiv.org/abs/1901.10430) |
| Routing | None | No routing mechanism | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | ? |
|  | | | | || |  |  |  |
| Position Embedding Position | Position Infused Attention | Bias (same shape as input embedding) added to queries | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Shortformer: Better Language Modeling using Shorter Inputs](https://arxiv.org/abs/2012.15832) |
| Position Embedding Position | Input Embedding | Bias added to embedded input | ? | ? | ? | ✔️️️️️️ | ❌️️ |  ❌️ | [Attention is all you need](https://arxiv.org/abs/1706.03762) |
| Position Embedding Position | None | No embedding | ? | ? | ? | ❌️️ | ✔️️ |  ❌️ | [Linear Transformers Are Secretly Fast Weight Memory Systems](https://arxiv.org/abs/2102.11174) |
| Position Embedding Position | Disentangled Attention | Attend to position embedding and context separately | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/1901.10430) |
|  | | | | || |  |  |  |
| Position Embedding Type | Rotary Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Relative Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Relative Learned Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Relative Learned Shared Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Axial Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Axial Shared Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Absolute Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
| Position Embedding Type | Absolute Shared Embedding | | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | |
|  | | | | || |  |  |  |
| Linear | More feedforward per attention | Use more feed forward layers before each attention block to produce better queries, keys, and values | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [DeLighT: Deep and Light-weight Transformer](https://arxiv.org/abs/2008.00623) |
| Linear | Group Linear Layer | Use group-linear-layers to reduce the computation cost of feed forward blocks while increasing their size | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://arxiv.org/abs/2006.11316) |
| Linear | Bottleneck multiplier | A stronger bottleneck reduces consumed time, memory, and parameters, while regularizing the model | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Bottleneck Transformers for Visual Recognition](https://arxiv.org/abs/2101.11605) |
| Linear | All-attention | Merged feed-forward with attention into one efficient block | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Augmenting Self-attention with Persistent Memory](https://arxiv.org/abs/1907.01470v1) |
| Linear | Layer Ordering | Perhaps attn-attn-ff-ff performs much better? | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [Improving Transformer Models by Reordering their Sublayers](https://arxiv.org/abs/1911.03864) |
|  | | | | || |  |  |  |
| Sequential | none | Default torch.nn.Sequential model without special layers | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [PyTorch Docs - Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential) |
| Sequential | RevNet | Memory efficient network | ? | ? | ? | ❌️️ | ✔️️ |  ❌️ | [MemCNN](https://github.com/silvandeleemput/memcnn) |
| Sequential | MomentumNet | Another memory efficient network | ? | ? | ? | ❌️️ | ✔️️ |  ❌️ | [MemCNN](https://github.com/silvandeleemput/memcnn) |
| Sequential | OmniNet | Every layer attends to all previous layers | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [OmniNet: Omnidirectional Representations from Transformers](https://arxiv.org/abs/2103.01075) |
| Sequential | DenseNet | Every layer uses features of all previous layers in feed-forwards | ? | ? | ? | ❌ | ❌️️ |  ❌️ | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) |
| Sequential | ResNet | Output of layer gets added to input | ? | ? | ? | ✔️️ | ❌️️ |  ❌️ | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| Sequential | ShuffleNet | Channel-shuffle connects to log2(features) previous layers | ? | ? | ? | ❌️️ | ❌️️ |  ❌️ | [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) |
|  | | | | || |  |  |  |
| Normalization Method | LayerNorm | Normalize and affine transform features of each instance | ? | ? | ? | ✔️️️️️️ | ✔️️️️ |  ❌️ | [Layer Normalization](https://arxiv.org/abs/1607.06450) |
| Normalization Method | RMSNorm | Normalize std and scale without mean norm/shift for features of each instance | ? | ? | ? | ❌️ | ❌️️️️️ |  ❌️ | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |
| Normalization Method | InstanceNorm | Normalize std and mean without scale/shift for features of each instance | ? | ? | ? | ❌️️️️️️️ | ❌️ |  ❌️ | [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) |
| Normalization Method | None | No normalization | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
| Normalization Method | SpaceNorm | LayerNorm but, the surrounding tokens are considered in mean/std | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
|  | | | | || |  |  |  |
| Normalization Type | Normalization | Default (LayerNorm). Ensures std/mean is constant at desired value | ? | ? | ? | ✔️️️️️️️ | ❌️️️️️ |  ❌️ |  |
| Normalization Type | ReZero | Multiply learnable scalar (initialized to 0) to layer outputs before applying residual | ? | ? | ? | ❌️️️️️️️ | ✔️️️️️ |  ❌️ | [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887) |
| Normalization Type | Weight Centralization | Remove output shift on the weight level by centralizing them at every step | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Weight and Gradient Centralization in Deep Neural Networks](https://arxiv.org/abs/2010.00866) |
| Normalization Type | Gradient Centralization | Remove weight shift on the gradient level by centralizing them | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461) |
| Normalization Type | Weight Standardisation | Remove output shift and scale on the weight level by normalizing them | ? | ? | ? | ❌️️️️️️️ | ❌  |  ❌️ | [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171) |
|  | | | | || |  |  |  |
| Optimizer | Adam | Default optimizer. Quite stable, quite expensive. Approximates 2nd order updates using two momentum buffers | ? | ? | ? | ✔️️️️️️️ | ❌️️️️️ |  ❌️ | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) |
| Optimizer | AdamW | Adam, but without broken weight decay | ? | ? | ? | ❌️️️️️️️ | ✔️️️️️ |  ❌️ |[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) |
| Optimizer | Shampoo | True 2nd order optimizer. Very stable, very fast (convergence/steps), extremely expensive. | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ |  [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568) |
| Optimizer | AdaHessian | True 2nd order optimizer with working implementation. Very stable, very fast, immeasurably expensive (memory, steps/s). | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719) |
| Optimizer | SM3 | Approximation of Adam. Faster with less memory consumption but worse convergence. | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150) |
| Optimizer | NovoGrad | Approximation of Adam. Faster with less memory consumption but worse convergence (they claim to converge better, but it couldn't be reproduced). | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks](https://arxiv.org/abs/1905.11286) |
| Optimizer | SMORMS3 | Adam variant with all fancy boltons like rectification (RAdam) | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
| Optimizer | MADGRAD | | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
|  | | | | || |  |  |  |
| Initialization | Normal 0.02 | Normal initialization with a small standard deviation is a naive default used by some people | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ |  |
| Initialization | Scaled Normal 0.02 | We scale the weights of residual layers at initial-ization by a factor of 1/√N where N is the number of residual layers. | ? | ? | ? | ✔️️️️️️️ | ❌️️️️️ |  ❌️ | |
| Initialization | Orthogonal | Random initialization with initial qr factorization as regularization | ? | ? | ? | ❌️️️️️️️ | ✔️️️️️ |  ❌️ | [Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks](https://arxiv.org/abs/2001.05992)|
| Initialization | Scaled Orthogonal |  | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | |
| Initialization | LSUV | Adaptive initialization based on network and its gradients | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [All you need is a good init](https://arxiv.org/abs/1511.06422)|
| Initialization | GradInit | New, adaptive initialization based on network and its gradients | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training](https://arxiv.org/abs/2102.08098)|
| Initialization | Kaiming | Normal initialization with calculated gain | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)|
|  | | | | || |  |  |  |
| Gradient Clipping | by value | clip gradients by their value to ensure NaNs don't make it through | ? | ? | ? | ✔️️️️️️️ | ❌️️️️️ |  ❌️ | |
| Gradient Clipping | by norm | clip gradients by a global norm to ensure the global update doesn't go above a specific range | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
| Gradient Clipping | adaptive gradient clipping | clip gradients by their norm relative to their weight norm to ensure weight updates won't disturb the weight | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171) |
| Gradient Clipping | none | no gradient clipping, to avoid filtering out valuable information | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
|  | | | | || |  |  |  |
| Input Tokenizer | Byte-Pair Encoding | encode pairs of arbitrary bytes to tokens, determined by relative frequency  | ? | ? | ? | ❌️️️️️️️ | ❌️️️️️ |  ❌️ | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)|
| Input Tokenizer | Optimized BPE | BPE, but with 65536 tokens. Allows to reduce padding (TPU) while making splitting more possible | ? | ? | ? | ❌️️️️️️️️️️️️ | ❌️️️️️ |  ❌️ | ? |
| Input Tokenizer | Character | avoid post-processing and model conditioning by feeding in true bytes | ? | ? | ? | ❌️️️️️️️ | ✔️️️️️ |  ❌️ | [CharBERT: Character-aware Pre-trained Language Model](https://arxiv.org/abs/2011.01513) |
|  | | | | || |  |  |  |
| Activation | SiLU | x * sigmoid(x) | ? | ? | ? | ❌️️️️️️️ | ❌ |  ❌️ | [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) |
| Activation | GELU | x * Φ(x) | ? | ? | ? | ✔️️️️️️️ | ️️️️️❌ |  ❌️ | [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) |
| Activation | Lecun TanH | x * tanh(x) + 0.1 | ? | ? | ️️️️️❌ | ️️️️️❌ | |
| Activation | Softsign | x / (abs(x) + 1) | ? | ? | ️️️️️❌ | ️️️️️❌ | |
| Activation | Mish | x * tanh(ln(e^x + 1)) | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️ |  ❌️ | [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681) |
| Activation | ReLU | (abs(x) + x)/2 | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ | [Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf) |
|  | | | | || |  |  |  |
| Activation Modifier | Normalization | norm(activate(f(x))), better generalization in cnns | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ |  |
| Activation Modifier | Dropout | dropout(activate(f(x))), apply small (0.2) dropout to feed forward's to avoid overfitting | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ ||
| Activation Modifier | GLU | activate(f(x)) * sigmoid(g(x)), gated linear unit| ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ |  |
| Activation Modifier | GLU_Add | activate(f(x)) * sigmoid(g(x)) + activate(h(x)) | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ | |
|  | | | | || |  |  |  |
| Training Mode | Masked Language | Simply mask input tokens (15%/25%/50%) | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ | |
| Training Mode | Autoregressive | Shift output by 1 token and use casual masking  | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ | |
| Training Mode | Query | Take known tokens and positions as one input to attend over and generate outputs using cross-attention | ? | ? | ? | ❌️️️️️️️ | ️️️️️✔️️️️️️️ |  ❌️ | |
|  | | | | || |  |  |  |