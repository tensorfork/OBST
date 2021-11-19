# OBST
**O**ptimal **B**are-minimum **S**ynchronisation **T**ransformer

Copyright (c) 2020-2021 The JanNet developers.

# Structure
* Abstract
  * Highly-efficent
  * Model-parallel
  * Perfomance boosts (with numbers)
* Introduction
  * Problem
    * Large transformers can't be trained on one device because of their parameter count
    * Model-parallel as alternative
    * Naive model-parallelism is highly inefficient
  * Current solution
    * other work recognized communication overhead being an issue
    * megatron tried to reduce that 
    * switch tried to get around the communcation overhead by doing more on each device using sparsity

  * "in the following we will describe what we do differently from megatron to achieve 8 billion parameters on a v3-8"
* Architecture
  * Reduction of communication overhead
    * Attention maps are always shared if _parameters_ are split across devices
      * Always split keys by heads instead of reshaping at all attention layers
      * Custom einsum required for dense layer
      * Splittig head dimension onto multiple devices possible
    * Fully-replicated dimension is required 
      * anonymize(x, dim) is about as fast as linear(x, [dim, anonymize_dim(dim)])
      * Feed forward with fully-replicated intermediate dim instead of linear projection
    * All-reduce is costly for large intermediate dimensions
      * reduce number of all-reduces by sharing the intermediate representation for qkv projections
      * reduce size of all-reduced dimension by a constant factor without performance drop (huge memory saving)
      * group-feed-forward interleaved with feed-forward and attention to avoid all-reduce altogether (see delight 
        for multiple ff's per attention)
  * Reduce slow pointwise 
    * Attention matrices are bottleneck
      * use jax softmax as it's faster with the same stability
      * scale key instead of the attention matrix if it's smaller
    * Normalization has too many pointwise ops + all-reduce
      * use rezero with high weight decay 
      * converges a bit slower but more stable and faster run time
    * Activation function
      * not lisht or mish
      * still not sure
      * either swish or relu
    * No embedding (it doesn't perform better per parameter)  
  * High memory consumption
    * RevNet
      * feed-forward, group-feed-forward and attention in their own revblocks
      * Input and output only have a simple linear layer
      * Embedding as bias term (same cost but higher reward)
    * Optimize optimizer
      * NovoGrad has layerwise 2nd moment -> less memory + compute
      * NovoGrad probably does something
      * Maybe fix it to work with matrices that have 1bln weights?
    * Custom crossentropy implementation
      * basically chunking, but for cross-entropy
      * more stable as it's avoiding numbers that blow up
      * optimized for mean reduction
* Loss plots
  * It's pretty fukn good on unseen (+ shuffled) data from ThePile 
  * Comparison to switch, megatron and gpt2
  * Do some fine-tuning if there is time left
  * Steal bmk's code to test it on lambada
* Conclusion
  * (Efficient) Model-parallelism is all you need
  * No, this model is not public
  * No, the weights are not public
  * No, this is not AGI
    
## Introduction

In recent years, transformers have gained a lot in popularity. With the research community catching on to their immense 
power and scaling capacity, countless experiments have been done to test their scaling laws and limits. So far, there
has only been a log-linear connection between the number of parameters and the evaluation loss. No limits were found 
yet. That's why recent transformers, such as GPT-3, GShard, and Switch-Tranformer were trained on 175B (dense) 
Parameters and 1T (sparse) Parameters.\
These models however can't be trained on a single device, which implies that the model, with all its weights, has to be
sharded across multiple GPUs to even fit a single batch. In the case of GShard and Switch-Transformer, 2048 TPUv3 
devices were used to train just one model. While the smaller models (32 devices) were trained for just 10 days, the 
biggest ones would likely need fractional more time per step.\
As we continue scaling our models, going beyond one trillion dense parameters, we will need to employ new techniques to
avoid training for ever-increasing times.

Other work, such as [Megatron-LM](https://arxiv.org/abs/1909.08053), [GShard](https://arxiv.org/abs/2006.16668), and
[Switch-Transformers](https://arxiv.org/abs/2101.03961), already recognized the high cost impact of data movement and 
sharding and attempted to reduce that. Megatron-LM scarcely used all-reduce operations during feed-forward blocks of 
the transformer, GShard further combined pipeline parallelism with a sparse 
["Mixture of Experts"](https://arxiv.org/abs/1701.06538) model, which Switch-Transformers later optimized using their 
own expert gating mechanism.\
In the following, we build on these works, and further improve upon them, to finally train GPT-3 scale models on a
single 32-core TPU.

## Architecture

We attempted to first reduce redundant computation, to ensure we won't be fooled by highly redundant over-utilization of
resources.\
In doing so, we noted that in the default attention model-parallel setting, attention maps are always shared across all
devices if the parameters of feed forward layers are split. This not only means that there are two necessary reshape
operations, directly after a matmul, per attention layer. It also means that both memory and computation are hugely
redundant as multiple, if not all, devices calculate the same attention maps in parallel.\
To avoid these costs, we decided to integrate the reshape operation into the matrix multiplication by replacing it with
an einsum of the input with `Shape[Batch, Sequence, Features]` and a parameter tensor with 
`Shape[Features, Heads, FeaturesPerHead]`. This allows us to split our parameter matrices across the head dimension
instead of the feature dimension, meaning that the attention matrices are calculated simultaneously on all devices while
still partitioning the parameters across model-parallel devices.
Of course, this only splits the attention layer. As feed-forward and normalization layers have to be split as well, we
decided to reshape the input/output tensor from `Shape[Batch, Sequence, Features]` to 
`Shape[Batch, Sequence, Heads, FeaturesPerHead]` in the entire model. While making normalization parallelism almost 
trivial, this change also makes feed-forward layers much harder to implement. Instead of relying on the framework 
([Mesh Tensorflow](https://github.com/tensorflow/mesh/)) to do all-reduces, a custom all-concat has to be implemented 
and called.\
Unfortunately, all-concat is significantly slower than all-sum, as it can't be parallelized as well by the backend 
framework ([Tensorflow](https://www.tensorflow.org/)). In fact, all-concat is so much slower that it is faster to insert 
a shared intermediate dimension using another matrix multiplication. Therefore, we used two linear layers in both 
feed-forward and attention.\
Instead of first reshaping `Shape[Batch, Sequence, Heads, FeaturesPerHead]` to 
`Shape[Batch, Sequence, FullyReplicatedHeads, FeaturesPerHead]` just to einsum it back to 
`Shape[Batch, Sequence, Heads, FeaturesPerHead]`, we decided to einsum from
`Shape[Batch, Sequence, Heads, FeaturesPerHead]`
to `Shape[Batch, Sequence, FullyReplicatedFeatures]` and back to `Shape[Batch, Sequence, Heads, FeaturesPerHead]`. Obviously
an activation is placed between these to maximally utilize the gained capacity.\
Later on however, we decided to give up the speed of the all-sum mentioned above to reduce memory consumption and 
number of redundant operations of our feed forward layers. The feed-forward block that's currently used first performs a
linear projection, followed by an activation and another linear project. The twist however is that the last projection
adds custom communication. For every head, we shift the activated input to the target head, perform our linear operation
as in the input, and finally sum all intermediate tensors. While this launches many small matrix multiplications and 
performs very suboptimal communication, it also reduces the memory consumption significantly and allows much more sparse 
connections. By only connecting each head to its left `sqrt(headss)` neighbors (rolling, of course), the communication 
overhead can be further reduced.\
As a next step against costly communication operations, we implemented group-feed-forward, group-instance-normalization,
and increased the number of group-feed-forward layers per attention. While this change doesn't impact model performance
at all, as it still has all the communication it needs in the feed forward coming from the attention layer, it increased
model speed by 10 to 50%, depending on the configuration. With only the changes above, we are now at roughly 60% MXU
utilization with the same configuration as the [baseline](https://github.com/EleutherAI/GPTNeo), which sits at a 
comfortable 2.5% MXU (which already is quite respectable, considering that keras struggles to achieve that without
model-parallelism).

After removing almost all communication overhead, the tensorboard profiler indicated that there is an excessive amount
of pointwise operations. With 20 to 70% of the runtime being consumed by ops like relu and addition, something had to 
be done.\
Since we have no redundant pointwise ops, we decided to go through all used high-level functions of the framework to 
optimize those.\
The only functions we're calling are softmax and cross-entropy, where cross-entropy is already optimal according to our
analysis. Softmax however uses `result=exp(x - logsumexp(x, dim))`, which is the default in many common frameworks, but 
significantly slower than the reference implementation in
[JAX](https://github.com/google/jax/blob/067be89a0c0e0650fc97d85f0ee11ff6588170a0/jax/_src/nn/functions.py#L297-L298). 
Unlike [PyTorch](https://pytorch.org/) and TensorFlow, JAX uses 
`tmp=exp(x - max(stopgrad(x), dim)); result=tmp / sum(tmp, dim)`, which avoids the logarithm, and the second
exponentiation. By adding this, we increased the MXU utilization to 66%, while keeping the pointwise ops at 10 to 50%.\
As a next step, we tried to optimize our graph by replacing operations with approximations. By doing so, we opened 
ourselves up to using [ReZero](https://arxiv.org/abs/2003.04887) as a post-norm instead of 
[Instance Normalization](https://arxiv.org/abs/1607.08022) as a post-norm.\
ReZero further bumps up the MXU utilization while also removing a few pointwise operations, as the model suddenly 
doesn't have to compute mean and variance anymore, but instead simply multiplies the output by a zero-initialized 
"alpha" parameter.\
Another nice thing about ReZero is that, once the first few layers have an alpha other than 0, ReZero converges much 
longer and to better minima than post-norm, as it is done in almost all recent work on transformers.\
However, to avoid the initial slowdown and final divergence, we decided to add group-instance-norm as a pre-norm while 
retaining ReZero as post-norm. This normalization configuration resulted in the most stable configuration that 
consistently achieved the lowest losses while also converging in less time than all others that were tested.\
A member of [EleutherAI (Leo Gao)](https://twitter.com/nabla_theta) also tested all known parametric and non-parametric
activation functions (~50 in total) on ThePile, a 1 TB plain text dataset. The result of said tests was that softsign
and ReLU perform the best, while ReLU also is the fastest activation of all. Thanks to these findings, we switched from
SiLU and LiSHT, to ReLU, which not only reduced the time spent on pointwise operations but also improved our 
convergence speed and accuracy.\
Lastly, we experimented with the removal of embeddings. Many papers such as
[Language Modeling with Deep Transformers](https://arxiv.org/abs/1905.04226) and 
[Linear Transformers Are Secretly Fast Weight Memory Systems](https://arxiv.org/abs/2102.11174) proposed to do so in 
the context of autoregressive language models. While we couldn't confirm their results, and strongly believe that
positional embedding helps the model, by reducing the loss by another 5% compared to not using it, it also decreases
the model speed by roughly ten percent, which is why we decided to remove it to further improve efficiency.

Unfortunately, even after all this, the memory consumption of the model still is unacceptably high.\
We try to work around this problem by first wrapping all blocks in a [RevNet](https://arxiv.org/abs/1707.04585) 
structure, like [Reformer](https://arxiv.org/abs/2001.04451) did. This way, only the activations (and attention 
matrices) of one "block" have to be stored, instead of all of them. In our case, a "block" consists of two functions
`f(x) = rezero(feedforward(norm(x)))` and `g(x) = rezero(attention(norm(x)))` (see the RevNet paper for more details 
on why you need two functions, instead of one). As both functions still have to be stored exactly once, it is of 
utmost interest to reduce the size of each of these. To do so, we broke down the default attention block from 
`f(x) = norm(attention(norm(feedforward(x))))` into its parts while also adding pre-norm as discussed above. This way,
the forward pass of the model only takes minimal amounts of memory, meaning that almost all memory is taken up by 
model parameters and other constant buffers.\
What does take a lot of memory however is the optimizer state. Adam uses two buffers of the size of the parameters, 
while Ranger uses three. Therefore, in the case of Ranger, 3/4 of the total memory consumption comes from the optimizer 
alone. As this isn't acceptable, we worked with and tested many other optimizers such as Novograd, AdaFactor, and SM3.
Unlike Adam, Novograd uses only one buffer, meaning that only half the memory is used to store the optimizer while 
converging as well as Adam. Novograd also has the additional advantage that it can better utilize large learning rates 
than Adam does. However, as it still uses significant amounts of memory, we further tested AdaFactor, which converged
very slowly to significantly worse minima than Adam, and SM3. SM3 on the other hand converges only a little worse than 
Adam, while reaching the same minima and using one millionth of the memory. \
With that, the model uses one set of buffers to store the parameters, a smaller buffer to store the per-block 
activations, and a final set of buffers to store the gradients of the parameters before applying them. To optimize the 
last part, we decided to re-implement the backward pass of Mesh Tensorflow and fuse it with our optimizer, by applying 
gradients on-the-fly instead of storing them. This way we managed to further reduce the memory consumption of the 
backward pass, to only store gradients currently necessary to backprop futher instead of all gradients.

Finally, we managed to increase the speed of the model roughly 100x while also linearizing the memory consumption that 
is already reduced linearly with the number of devices. With that, we can scale our significantly faster and more 
efficient model almost linearly by simply adding more devices, without falling back to sparsity.