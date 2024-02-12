"""
Train the transformer on a translation task using the opus_books dataset:
https://huggingface.co/datasets/opus_books
"""
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data import BilingualDataset, causal_mask
from model import Transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers, decoders
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer  # Train the tokenizer given the list of sentences
from tokenizers.pre_tokenizers import Whitespace, Digits
from pathlib import Path

from tqdm import tqdm


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):

    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        # tokenizer.pre_tokenizers = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        # tokenizer.decoder = decoders.WordPiece()

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        tokenizer.decoder = decoders.WordPiece()

        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'])
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):

    dataset_raw = load_dataset(
        'opus_books',
        f'{config["source_language"]}-{config["target_language"]}',
        split='train'
    )

    # TODO: remove data limit
    dataset_raw = torch.utils.data.Subset(dataset_raw, indices=range(100))

    # Build tokenizers
    source_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['target_language'])

    # Split dataset into train and validation
    train_dataset_size = int(config['train_split_percent'] * len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    # Create the dataset (access the tensors)
    train_ds = BilingualDataset(
        train_dataset_raw,
        source_tokenizer,
        target_tokenizer,
        config['source_language'],
        config['target_language'],
        config['sequence_len']
    )
    val_ds = BilingualDataset(
        train_dataset_raw,
        source_tokenizer,
        target_tokenizer,
        config['source_language'],
        config['target_language'],
        config['sequence_len']
    )

    max_source_len = 0
    max_target_len = 0

    for item in dataset_raw:
        source_ids = source_tokenizer.encode(item['translation'][config['source_language']]).ids
        target_ids = source_tokenizer.encode(item['translation'][config['target_language']]).ids
        max_source_len = max(max_source_len, len(source_ids))
        max_target_len = max(max_target_len, len(target_ids))

    print(f'Max length of source sentence: {max_source_len}')
    print(f'Max length of target sentence: {max_target_len}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)  # batch==1 to process each sentence one by one

    return train_dataloader, val_dataloader, source_tokenizer, target_tokenizer


def get_model(config, source_vocab_size, target_vocab_size):
    model = Transformer.build(
        source_vocab_size,
        target_vocab_size,
        config['sequence_len'],
        config['sequence_len'],
        config['embedding_size'],
        heads=config['heads']
    )
    return model


def train_model(config):

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, source_tokenizer, target_tokenizer = get_dataset(config)
    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size())

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloding model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input']  # (batch, seq_len)
            decoder_input = batch['decoder_input']  # (batch, seq_len)
            encoder_mask = batch['encoder_mask']  # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask']  # (batch, 1, seq_len, seq_len)

            # Run the tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (batch, seq_len, embedding_size)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batch, seq_len, embedding_size)
            projection_output = model.project(decoder_output)  # (batch, seq_len, target_vocab_size)

            # Compare to the label and compute loss
            label = batch['label']  # (batch, seq_len)
            loss = loss_fn(projection_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            # Log
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropage the loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
