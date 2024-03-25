# Transformer

Pytorch lightning implementation of the original Transformer architecture 
as described in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper.
Along with the architecture, the repo contains the code to run training and inference on
a machine translation task to translate from english to italian. A Tokenizer and a Dataloader are provided as well. 
The dataloader uses the [OPUS Books](https://huggingface.co/datasets/opus_books) dataset.
