# BPE Tokenizer
## Status Quo
Right now, June 3rd, 2021, the GPT-2 tokenizer is the standard tokenizer used by all major GPT projects such as
[NeoX](https://github.com/EleutherAI/gpt-neox),
[Mesh Transformer Jax](https://github.com/kingoflolz/mesh-transformer-jax) and
[OBST](https://github.com/ClashLuke/JanNet/). The GPT-2 tokenizer is a default
[HuggingFace BPE tokenizer](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.models.BPE)
with 50257 tokens containing 50000 trained tokens, the 256 first unicode symbols and a special `<|endoftext|>` token.
Alternatives to this exist, such as BERT's WordPiece which has roughly 30000 tokens and 994 reserved unused tokens.\
While this is going on the open-source sphere, OpenAI silently  