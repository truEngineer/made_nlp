import re
import os
import random
import string
from typing import Optional, Pattern

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

TOKEN_RE = re.compile(r'-?\d*\.\d+|[a-zа-яё]+|-?\d+|\S', re.I)  # ignore case


def tokenize_text_regex(txt: str, regex: Pattern[str], min_token_size: int = 0) -> list[str]:
    """Tokenize text with regex
    Args:
        txt: text to tokenize
        regex: re.compile output
        min_token_size: min char length to highlight as token
    Returns:
        tokens list
    """
    all_tokens = regex.findall(txt.lower())
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize(corpus: list[str]) -> list[list[str]]:
    """Tokenize text corpus with simple regex
    Args:
        corpus: text corpus
    Returns:
        List of tokenized texts
    """
    tokenized_corpus = []
    for doc in corpus:
        tokenized_corpus.append(tokenize_text_regex(doc, TOKEN_RE))

    return tokenized_corpus


def make_labeling(tokenized_corpus: list[list[str]], save_path: Optional[str] = None) -> list[list[str]]:
    """
    Make labeling to correspond BertPunc input data https://github.com/IsaacChanghau/neural_sequence_labeling/tree/master/data/raw/LREC
    Args:
        tokenized_corpus: tokenized text corpus
        save_path: path to save labeling result
    Returns:
        labeled tokenized text corpus
    """
    labeled_tokens = []
    for text_tokenized in tokenized_corpus:
        text_tokenized.append("")
        for i in range(len(text_tokenized) - 1):
            if text_tokenized[i] in string.punctuation:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens[-1][1] = "PERIOD"
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens[-1][1] = "COMMA"
                elif text_tokenized[i + 1] == "?":
                    labeled_tokens[-1][1] = "QUESTION"
                else:
                    continue
            else:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens.append([text_tokenized[i], "PERIOD"])
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens.append([text_tokenized[i], "COMMA"])
                elif text_tokenized[i + 1] == "?":
                    labeled_tokens.append([text_tokenized[i], "QUESTION"])
                else:
                    labeled_tokens.append([text_tokenized[i], "OVERALL"])

    if save_path is not None:
        with open(save_path, "w") as f:
            for token, label in labeled_tokens:
                f.write(f"{token}\t{label}\n")

    return labeled_tokens


def make_datasets(
        path_to_preprocessed_corpus: str, config: dict
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """
    Create labelled train, valid, test datasets for BertPunc
    Args:
        path_to_preprocessed_corpus: path to preprocessed text corpus
        config: config
    Returns:
        (train corpus, valid corpus, test corpus)
    """
    with open(path_to_preprocessed_corpus) as f:
        corpus = f.readlines()

    train_corpus, valid_corpus = train_test_split(
        corpus,
        random_state=config['random_seed'],
        test_size=config['valid_rate'] + config['test_rate'],
    )
    valid_corpus, test_corpus = train_test_split(
        valid_corpus,
        random_state=config['random_seed'],
        test_size=config['test_rate'],
    )

    train_corpus = tokenize(train_corpus)
    valid_corpus = tokenize(valid_corpus)
    test_corpus = tokenize(test_corpus)

    train_corpus = make_labeling(train_corpus, config['train_path_name'])
    valid_corpus = make_labeling(valid_corpus, config['valid_path_name'])
    test_corpus = make_labeling(test_corpus, config['test_path_name'])

    print(f"Train amount: {len(train_corpus)}\nValid amount: {len(valid_corpus)}\nTest amount: {len(test_corpus)}")

    return train_corpus, valid_corpus, test_corpus


def init_random_seed_torch(value: int = 0) -> None:
    """Initializes random seed for reproducibility random processes
    Args:
        value:
    Returns:
        None
    """
    random.seed(value)
    np.random.seed(value)
    os.environ['PYTHONHASHSEED'] = str(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


# def gpus_to_use(gpus_list: Optional[list] = None) -> None:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     if gpus_list is not None:
#         gpus_list = [str(item) for item in gpus_list]
#         gpus = ",".join(gpus_list)
#         os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpus}"
#     else:
#         os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
#
# def get_device(devices_list: Optional[list] = None) -> torch.device:
#     if torch.cuda.is_available():
#         gpus_to_use(devices_list)
#         return torch.device("cuda")
#     else:
#         return torch.device("cpu")


def load_file(filename: str) -> list[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data


def encode_data(data: list[str], tokenizer: PreTrainedTokenizer, punctuation_enc: dict) -> tuple[list, list]:
    """
    Converts words to (BERT) tokens and punctuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X, Y = [], []
    for line in data:
        word, punc = line.split('\t')
        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y


def insert_target(x, segment_size) -> np.ndarray:
    """
    Creates segments of surrounding words for each word in x.
    Inserts a zero token halfway the segment.
    """
    X = []
    x_pad = x[-((segment_size-1)//2-1):] + x + x[:segment_size//2]

    for i in range(len(x_pad)-segment_size+2):
        segment = x_pad[i:i+segment_size-1]
        segment.insert((segment_size-1)//2, 0)
        X.append(segment)

    return np.array(X)


def preprocess_data(
        data: list, tokenizer: PreTrainedTokenizer, punctuation_enc: dict, segment_size: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y = encode_data(data, tokenizer, punctuation_enc)
    X = insert_target(X, segment_size)
    return X, np.array(y)


def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader
