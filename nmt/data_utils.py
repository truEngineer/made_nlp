import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import WordPunctTokenizer
from transformers import AutoTokenizer


en_tokenizer = WordPunctTokenizer()
ru_tokenizer = WordPunctTokenizer()
en_bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")  # bert-base-uncased
# AutoTokenizer.from_pretrained("distilbert-base-cased", bos_token='[BOS]', eos_token='[EOS]')
# Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.


def yield_tokens(data_path, lang='en'):
    with open(data_path, encoding='utf8') as f:
        for line in f:
            en_line, ru_line = line.split('\t')
            # en: 'Cordelia Hotel is situated in Tbilisi, a 3-minute walk away from Saint Trinity Church.'
            # ru: 'Отель Cordelia расположен в Тбилиси, в 3 минутах ходьбы от Свято-Троицкого собора.\n'
            if lang == 'en':
                yield en_tokenizer.tokenize(en_line.lower())
            elif lang == 'ru':
                yield ru_tokenizer.tokenize(ru_line[:-1].lower())  # remove newline character


def get_vocab(data_path, lang):
    lang_vocab = build_vocab_from_iterator(
        yield_tokens(data_path, lang),
        specials=['<unk>', '<pad>', '<bos>', '<eos>'],  # Special symbols to add. The order will be preserved.
        min_freq=3,  # Tokens that appear only once are converted into an <unk> (unknown) token.
    )
    lang_vocab.set_default_index(lang_vocab['<unk>'])
    # Value of default index. This index will be returned when OOV token is queried
    # Out of Vocabulary (OOV)
    return lang_vocab


def process_data(data_path):
    en_vocab = get_vocab(data_path, lang='en')
    ru_vocab = get_vocab(data_path, lang='ru')

    data = []
    with open(data_path, encoding='utf8') as f:
        for line in f:
            en_line, ru_line = line.split('\t')
            en_tensor = torch.tensor(
                [en_vocab[token] for token in en_tokenizer.tokenize(en_line.lower())], dtype=torch.long)
            ru_tensor = torch.tensor(
                [ru_vocab[token] for token in ru_tokenizer.tokenize(ru_line[:-1].lower())], dtype=torch.long)
            data.append((en_tensor, ru_tensor))

    return data, en_vocab, ru_vocab


class En2RuDataset(Dataset):
    def __init__(self, en_ru_tensors):
        self.en_ru_tensors = en_ru_tensors

    def __getitem__(self, index):
        src_seq = self.en_ru_tensors[index][0]
        trg_seq = self.en_ru_tensors[index][1]
        return src_seq, trg_seq  # return sample

    def __len__(self):
        return len(self.en_ru_tensors)


def generate_batch(data_batch, en_vocab, ru_vocab, batch_first=False):
    pad_idx_en, pad_idx_ru = en_vocab['<pad>'], ru_vocab['<pad>']
    bos_idx_en, bos_idx_ru = en_vocab['<bos>'], ru_vocab['<bos>']
    eos_idx_en, eos_idx_ru = en_vocab['<eos>'], ru_vocab['<eos>']
    en_batch, ru_batch = [], []
    for (en_item, ru_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([bos_idx_en]), en_item, torch.tensor([eos_idx_en])], dim=0))
        ru_batch.append(torch.cat([torch.tensor([bos_idx_ru]), ru_item, torch.tensor([eos_idx_ru])], dim=0))
    # stack a list of tensors along a new dimension, and pad them to equal length
    en_batch = pad_sequence(en_batch, padding_value=pad_idx_en)
    ru_batch = pad_sequence(ru_batch, padding_value=pad_idx_ru)

    if batch_first:
        en_batch = en_batch.permute(1, 0)
        ru_batch = ru_batch.permute(1, 0)

    return en_batch, ru_batch


def process_data_bert(data_path):
    ru_vocab = get_vocab(data_path, lang='ru')

    data = []
    with open(data_path, encoding='utf8') as f:
        for line in f:
            en_line, ru_line = line.split('\t')
            # add_special_tokens=True, add [CLS] and [SEP] tokens (instead '[BOS]' and '[EOS]' tokens)
            en_tensor = torch.tensor(
                en_bert_tokenizer.encode(en_line.lower(), truncation=True, max_length=512,
                                         add_special_tokens=True), dtype=torch.long)
            ru_tensor = torch.tensor(
                [ru_vocab[token] for token in ru_tokenizer.tokenize(ru_line[:-1].lower())], dtype=torch.long)
            data.append((en_tensor, ru_tensor))

    return data, ru_vocab


def generate_batch_bert(data_batch, ru_vocab, batch_first=False):
    en_batch, ru_batch = [], []
    for (en_item, ru_item) in data_batch:
        en_batch.append(en_item)
        ru_batch.append(
            torch.cat([torch.tensor([ru_vocab['<bos>']]), ru_item, torch.tensor([ru_vocab['<eos>']])], dim=0))
    # stack a list of tensors along a new dimension, and pad them to equal length
    en_batch = pad_sequence(en_batch, padding_value=en_bert_tokenizer.pad_token_id)
    ru_batch = pad_sequence(ru_batch, padding_value=ru_vocab['<pad>'])

    if batch_first:
        en_batch = en_batch.permute(1, 0)
        ru_batch = ru_batch.permute(1, 0)

    return en_batch, ru_batch


def process_data_bert(data_path):
    ru_vocab = get_vocab(data_path, lang='ru')

    data = []
    with open(data_path, encoding='utf8') as f:
        for line in f:
            en_line, ru_line = line.split('\t')
            # add_special_tokens=True, add [CLS] and [SEP] tokens (instead '[BOS]' and '[EOS]' tokens)
            en_tensor = torch.tensor(
                en_bert_tokenizer.encode(en_line.lower(), truncation=True, max_length=512,
                                         add_special_tokens=True), dtype=torch.long)
            ru_tensor = torch.tensor(
                [ru_vocab[token] for token in ru_tokenizer.tokenize(ru_line[:-1].lower())], dtype=torch.long)
            data.append((en_tensor, ru_tensor))

    return data, ru_vocab


def generate_batch_bert(data_batch, ru_vocab, batch_first=False):
    en_batch, ru_batch = [], []
    for (en_item, ru_item) in data_batch:
        en_batch.append(en_item)
        ru_batch.append(
            torch.cat([torch.tensor([ru_vocab['<bos>']]), ru_item, torch.tensor([ru_vocab['<eos>']])], dim=0))
    # stack a list of tensors along a new dimension, and pad them to equal length
    en_batch = pad_sequence(en_batch, padding_value=en_bert_tokenizer.pad_token_id)
    ru_batch = pad_sequence(ru_batch, padding_value=ru_vocab['<pad>'])

    if batch_first:
        en_batch = en_batch.permute(1, 0)
        ru_batch = ru_batch.permute(1, 0)

    return en_batch, ru_batch
