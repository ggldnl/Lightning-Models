from models.transformer.model import Transformer
from tokenizer import WordLevelTokenizer
from data import OPUSDataModule
import pytorch_lightning as pl
import torch.nn as nn
import config
import os


def create_tokenizer(stage='source'):  # stage = ['source', 'target']

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
    corpus = []
    for batch in train_dataloader:
        # batch[f'{stage}_text'] is a list of strings
        corpus.extend(batch[f'{stage}_text'])

    # Use the corpus to train the tokenizer
    tokenizer = WordLevelTokenizer()
    tokenizer.train(corpus)

    return tokenizer


if __name__ == '__main__':

    # Configure logging level
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

    # Create or get tokenizers
    source_tokenizer_path = r'tokenizers/tokenizer_source.json'
    target_tokenizer_path = r'tokenizers/tokenizer_target.json'

    # Check if a tokenizer backup exists
    if os.path.exists(source_tokenizer_path):
        print(f'Loading source tokenizer...')
        source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path)
    # If not, create a monolingual dataset and train them
    else:
        print(f'Creating source tokenizer...')
        source_tokenizer = create_tokenizer('source')
        source_tokenizer.save(source_tokenizer_path)

    if os.path.exists(target_tokenizer_path):
        print(f'Loading target tokenizer...')
        target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path)
    else:
        print(f'Creating target tokenizer...')
        target_tokenizer = create_tokenizer('target')
        target_tokenizer.save(target_tokenizer_path)

    # Create a datamodule with the tokenizers
    datamodule = OPUSDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        download=False
    )

    # Create the model
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

    # Create the trainer
    trainer = pl.Trainer(
        min_epochs=1, max_epochs=config.NUM_EPOCHS,
        # precision=config.PRECISION,
        accelerator='auto'
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
