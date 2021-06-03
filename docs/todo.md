### Todo
#### Implement
##### Optimizer
- [ ] Shampoo optimizer
- [ ] ADAS
- [ ] MADGRAD
- [ ] cyclic learning rate
- [ ] momentum policy (https://twitter.com/_arohan_/status/1388886551304695808)
##### Model/Feature
- [ ] Pipeline parallelism
- [ ] spacenorm
- [ ] all-to-all and channel-shuffle as custom ops
- [ ] sparse lingvo-like mixture of experts
- [ ] dropconnect
- [ ] query layer (pangu-alpha) to optimize bert querying
##### Model/Performance
- [ ] split context on accelerator instead of data pipeline (-> less CPU memory, maybe faster?)
- [ ] optimize location of dimensions to reduce communication overhead
##### Training loop
- [ ] increasing the context size over time (shortformer)
- [ ] Put ops in scope / figure out what init_scope variable_ops scope etc do
- [ ] weight averaging for data-parallel (like in Shawn Presser's "Swarm Training")
#### Run
##### Model
- [ ] Large (200B) run
- [ ] test bert for next-token prediction
- [ ] adam hyperparameter search
- [ ] 300M samples to show aleph-alpha team
- [ ] adaptive gradient clipping experiments
##### Tokenizer
- [ ] human-in-the-loop evaluation of large char-level models
#### Feedback required
- [ ] release tokenizer? (if it performs better?) 
- [ ] release the cleaned variant of ThePile (ftfy + minor replacements)? 
#### Reading list
##### Papers
- [ ] https://arxiv.org/abs/2102.05610 (Searching for Fast Model Families on Datacenter Accelerators)
- [ ] https://arxiv.org/abs/2104.14830 (Scaling End-to-End Models for Large-Scale Multilingual ASR)
- [ ] https://arxiv.org/abs/2105.00572 (Larger-Scale Transformers for Multilingual Masked Language Modeling)
- [ ] https://arxiv.org/abs/1506.02142 (dropout as Bayesian)
- [ ] https://arxiv.org/abs/2010.15327 (Do Wide and Deep Networks Learn the Same Things?)
- [ ] https://arxiv.org/abs/2104.05158 (High-performance, Distributed Training of Large-scale Deep Learning Recommendation Models)
- [ ] https://arxiv.org/abs/2103.10427 (The Low-Rank Simplicity Bias in Deep Networks)
- [ ] https://arxiv.org/abs/2104.00298 (EfficientNetV2)
- [ ] https://arxiv.org/abs/2105.03824 (Fnet)
- [ ] https://arxiv.org/abs/2104.01136 (LeViT)
- [ ] https://arxiv.org/abs/2102.02611 (CKConv)
- [ ] https://arxiv.org/abs/2105.03322 (pretrained conv > attn?)
- [ ] https://arxiv.org/abs/2105.04551 (stochastic invertible network)
- [ ] https://arxiv.org/abs/2105.03928 (vocabulary size analysis in transformers)
- [ ] https://arxiv.org/abs/2105.04663 (large-scale parallelism+speed+engineering research)
- [ ] https://arxiv.org/abs/2105.04779 (exact, efficient attention)
- [ ] https://arxiv.org/abs/2104.05704 (compact transformer)
- [ ] https://arxiv.org/abs/2102.06356 (A Large Batch Optimizer Reality Check)
- [ ] https://arxiv.org/abs/2103.10360 (BERT = GPT)
- [ ] https://arxiv.org/abs/1708.07120 (superconvergence)
- [ ] https://arxiv.org/abs/2104.04473 (megatron2)
##### Books
- [ ] https://battle.shawwn.com/sdb/books/PSP/ fourier transformation something

### Done
- [x] one-cycle lr policy
- [x] reduce_lr_on_plateau
- [x] implement momentumnet (https://arxiv.org/abs/2102.07870)
- [x] log EMA(losses) instead or losses[-1] in macrobatch
- [x] implement MLP Mixer, gMLP
- [x] clean ThePile
- [x] train new tokenizer on TheCleanedPile (vocab=65536, split_at=string.puctionation + string.digits + string.whitespace)
- [x] Split vocab-dim by heads in input, output, and eval -> 15% faster with n_heads times less memory
- [x] Split video target by heads in input, output, and eval
- [x] Custom op to calculate softmax logits in one einsum -> not worth it
- [x] Custom op for group-linear with same output shape (-> not necessary)
- [x] Test all_mean (counterproductive)
- [x] Group-attention + feed-forward vs group-linear + attention (-> insignificant difference)
- [x] (Working) eval of small models -> Loss of 2.8 achieved with 100M model on ThePile within a day of training. samples look good, even in Russian, but aren't available anymore
- [x] Fix Adaptive Gradient Clipping
- [x] Readd clip_by_norm for gradient clipping (-> much more stable for relu, gelu performance is the same)
- [x] SMORMS3 optimizer (-> performs worse than baseline AdamW on CNNs. Highly unlikely transformers benefit from it)
- [x] [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/pdf/1810.04650.pdf) -> performs horribly (implemented correctly by Jan)
- [x] tensorflow compilation to tf.function (-> incompatible)
- [x] tf.data.Dataset to random shuffled queue (counterproductive as mp would be done by hand)
- [x] tanh(x) + 0.1x activation
- [x] Store weights on tpu cpu -> done by mtf already
- [x] (small-ctx) lightweight conv using addn -> involution performs better in CNNs and implemented, not experiments yet 
- [x] fix information leakage -> inf * 0 == NaN, but -1e32 is too small for the mask to prevent leakage
- [x] test openai softmax masking -> roughly 1% slower with slightly more memory usage than naive masking
- [x] Gradient checkpointing for input (superseded by "input as custom op")
- [x] Further investigate trace to figure out why it's idling so much (-> solved by macrobatching)
- [x] Sinusoidal posembd
- [x] Shared posembd
- [x] retokenize pile to skip more efficiently (number of tokens needed)
- [x] train new tokenizer
- [x] fix multi-device argmax
- [x] add dropout block
- [x] merge dropout into feed-forward (-> params object?)
- [x] read https://arxiv.org/abs/2001.08361
- [x] axial posembd
- [x] split posembd
- [x] learned relative posembd
- [x] additive posembd
- [x] stop_gradient (-> adds more memory consumption as it's still traversing the entire graph, just one op more now)
- [x] guarantee_const (-> same as for stop_gradient)
- [x] custom groupnorm op (requirement for memory alignment + faster + less mem)
- [x] custom grouplinear op (requirement for memory alignment)
- [x] pack many small (4bit/12bit) integers into one large integer to save communication space
- [x] improved memory alignment (-> more reshape, not worth it)
- [x] custom embed op (faster + less mem)
- [x] input as custom op (-> already cleaned up by tf)
- [x] merge attention into custom op to explicitly calculate gradient (given that the norm custom op doesn't improve memory and doesn't speed things up, it's not worth it, unless there's a huge, bad op, like in softmax)