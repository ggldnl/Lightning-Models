import os.path

from models.transformer.model import Transformer
from tokenizer import WordLevelTokenizer
from data import OPUSDataModule
import torch.nn as nn
import config


def create_tokenizers():

    # Build a datamodule WITHOUT tokenizers
    datamodule = OPUSDataModule(
        config.DATA_DIR,
        max_seq_len=config.MAX_SEQ_LEN,
        download=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Take the corpus (we need an iterable)
    train_dataloader = datamodule.train_dataloader()
    source_corpus = []
    target_corpus = []
    for batch in train_dataloader:
        # batch[f'{stage}_text'] is a list of strings
        source_corpus.extend(batch['source_text'])
        target_corpus.extend(batch['target_text'])

    # Print a dataset sample
    for i in range(3):
        print(f"Sentence pair [{i}]:\n\tsource: {source_corpus[i]}\n\ttarget: {target_corpus[i]}\n")

    # Use the corpus to train the tokenizers
    source_tokenizer = WordLevelTokenizer()
    source_tokenizer.train(source_corpus)

    target_tokenizer = WordLevelTokenizer()
    target_tokenizer.train(target_corpus)

    return source_tokenizer, target_tokenizer


if __name__ == '__main__':

    # Configure logging level
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

    # Create tokenizers
    print(f'Creating source and target tokenizers...')
    source_tokenizer, target_tokenizer = create_tokenizers()

    print(f'Source and target tokenizers created.')
    print(f'Source tokenizer vocabulary size: {source_tokenizer.vocab_size}')
    print(f'Target tokenizer vocabulary size: {target_tokenizer.vocab_size}')

    # Create the model or restore it from a checkpoint
    checkpoint_path = '/home/daniel/Git/Lightning-Models/models/transformer/machine_translation/lightning_logs/version_41/'
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint...')
        model = Transformer.load_from_checkpoint(checkpoint_path)
        print(f'Checkpoint successfully loaded')
    else:
        model = Transformer.build(
            source_vocab_size=source_tokenizer.vocab_size,
            target_vocab_size=target_tokenizer.vocab_size,
            source_sequence_length=config.MAX_SEQ_LEN,
            target_sequence_length=config.MAX_SEQ_LEN,
            learning_rate=config.LEARNING_RATE,
            embedding_size=config.EMBED_DIM,
            num_encoders=config.NUM_ENCODERS,
            num_decoders=config.NUM_DECODERS,
            dropout=config.DROPOUT,
            heads=config.HEADS,
            d_ff=config.D_FF,
            loss_fn=nn.CrossEntropyLoss(ignore_index=source_tokenizer.pad_token_id, label_smoothing=0.1)
        )

    input_text = 'Hello, I\'m a transformer.'
    tokenized_input = source_tokenizer.encode(input_text)

    print(f'Input sentence : {input_text}')
    print(f'Tokenized input: {tokenized_input}')

    model_output = model.translate(
        tokenized_input,
        max_output_length=20,
        target_sos_token_id=target_tokenizer.sos_token_id,
        target_eos_token_id=target_tokenizer.eos_token_id
    )
    decoded_output = target_tokenizer.decode(model_output)

    print(f'Model output   : {model_output}')
    print(f'Decoded output : {decoded_output}')
