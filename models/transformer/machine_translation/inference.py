import os.path

from models.transformer.model import Transformer
from tokenizer import WordLevelTokenizer
from data import OPUSDataModule
import torch.nn as nn
import config


"""
def create_tokenizers(source_tokenizer_path=None, target_tokenizer_path=None):

    # Build a datamodule WITHOUT tokenizers
    datamodule = OPUSDataModule(
        config.DATA_DIR,
        max_seq_len=config.MAX_SEQ_LEN,
        download=False,
        random_split=False
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
    if source_tokenizer_path is not None:
        ext = source_tokenizer_path.split('.')[-1]
        source_tokenizer.store(source_tokenizer_path, driver=ext)

    target_tokenizer = WordLevelTokenizer()
    target_tokenizer.train(target_corpus)
    if target_tokenizer_path is not None:
        ext = target_tokenizer_path.split('.')[-1]
        target_tokenizer.store(target_tokenizer_path, driver=ext)

    return source_tokenizer, target_tokenizer
"""


if __name__ == '__main__':

    # Configure logging level
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

    """
    # Create tokenizers
    print(f'Creating source and target tokenizers...')
    source_tokenizer, target_tokenizer = create_tokenizers()
    """

    # Load the tokenizers
    source_tokenizer_path = r'tokenizers/tokenizer_source.pkl'
    target_tokenizer_path = r'tokenizers/tokenizer_target.pkl'
    source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path, driver='pkl')
    target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path, driver='pkl')

    print(f'Source and target tokenizers created.')
    print(f'Source tokenizer vocabulary size: {source_tokenizer.vocab_size}')
    print(f'Target tokenizer vocabulary size: {target_tokenizer.vocab_size}')

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

    # Restore a checkpoint
    checkpoint_path = '/home/daniel/Git/Lightning-Models/models/transformer/machine_translation/lightning_logs/version_50/checkpoints/epoch=9-step=14020.ckpt'
    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint...')
        model.load_from_checkpoint(checkpoint_path)
        print(f'Checkpoint successfully loaded')

    input_text = 'I\'m a good boy.'

    print(f'Input sentence : {input_text}')
    print(f'Tokenized input: {source_tokenizer.encode(input_text)}')

    model_output = model.translate(
        input_text,
        source_tokenizer,
        target_tokenizer,
        max_output_length=20,
    )
    decoded_output = target_tokenizer.decode(model_output)

    print(f'Model output   : {model_output}')
    print(f'Decoded output : {decoded_output}')
