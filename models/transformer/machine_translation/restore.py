from models.transformer.model import Transformer
from tokenizer import WordLevelTokenizer
import torch.nn as nn
import config
import torch


if __name__ == '__main__':

    # Load the tokenizers
    source_tokenizer_path = r'tokenizers/tokenizer_source.pkl'
    target_tokenizer_path = r'tokenizers/tokenizer_target.pkl'
    source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path, driver='pkl')
    target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path, driver='pkl')

    # Restore state_dict
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
        loss_fn=nn.CrossEntropyLoss(ignore_index=source_tokenizer.pad_token_id, label_smoothing=0.1),
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer
    )
    checkpoint = torch.load("/home/daniel/Git/Lightning-Models/models/transformer/machine_translation/lightning_logs/version_0/checkpoints/")
    model.load_state_dict(checkpoint["state_dict"])

    """
    # Restore the whole model
    checkpoint = torch.load("/home/daniel/Git/Lightning-Models/models/transformer/machine_translation/lightning_logs/version_0/checkpoints/")
    model = Transformer.load_from_checkpoint(checkpoint)
    """

    # Inference
    input_text = 'Hello, alice, how are you?'
    model.translate(input_text, source_tokenizer, target_tokenizer, max_output_length=20)
