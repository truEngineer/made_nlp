import os
import json
import warnings
from datetime import datetime

import torch
from torch import nn, optim
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoTokenizer

from model import BertPunc
from utils import (make_datasets, init_random_seed_torch,
                   load_file, preprocess_data, create_data_loader)
from train import Trainer


if __name__ == "__main__":
    os.makedirs('dataset', exist_ok=True)
    dataset_config = {
        "random_seed": 78,
        "valid_rate": 0.2,
        "test_rate": 0.1,
        "train_path_name": "dataset/01_punct_pushkin_train.txt",
        "valid_path_name": "dataset/01_punct_pushkin_valid.txt",
        "test_path_name": "dataset/01_punct_pushkin_test.txt",
    }
    train_corpus, valid_corpus, test_corpus = make_datasets("./01_punct_pushkin.txt", dataset_config)

    punctuation_enc = {'OVERALL': 0, 'COMMA': 1, 'PERIOD': 2, 'QUESTION': 3}
    # punct_to_token = {
    #     ' ' : 'S',  # space
    #     ',' : 'C',  # comma
    #     '.' : 'P',  # period
    #     '!' : 'EX', # exclamation
    #     '?' : 'Q'   # question
    # }

    segment_size = 16
    epochs_top = 1
    iterations_top = 2
    batch_size_top = 512
    learning_rate_top = 1e-5
    epochs_all = 4
    iterations_all = 3
    batch_size_all = 256
    learning_rate_all = 1e-5
    dropout = 0.3

    train_config = {
        "segment_size": segment_size,
        "epochs_top": epochs_top,
        "iterations_top": iterations_top,
        "batch_size_top": batch_size_top,
        "learning_rate_top": learning_rate_top,
        "epochs_all": epochs_all,
        "iterations_all": iterations_all,
        "batch_size_all": batch_size_all,
        "learning_rate_all": learning_rate_all,
        "model": {
            "name_or_path": "DeepPavlov/rubert-base-cased-sentence",  # backbone Transformer-based model
            "tokenizer_vocab_size": None,
            "dropout": dropout,
            "mode": {
                "name": "stacked_hidden_states",
                "config": {
                    "type": "concat",
                    "n_layers": 4,
                    "reverse": True,
                    "sent_agg_type": "mean"
                }
            }
        }
    }

    data_train = load_file("dataset/01_punct_pushkin_train.txt")
    data_valid = load_file("dataset/01_punct_pushkin_valid.txt")
    print(f"Train data: {len(data_train)}, Valid data: {len(data_valid)}")

    tokenizer = AutoTokenizer.from_pretrained(train_config["model"]["name_or_path"], do_lower_case=True)
    train_config["model"]["tokenizer_vocab_size"] = tokenizer.vocab_size  # set tokenizer_vocab_size
    save_path = f"checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(save_path, exist_ok=True)

    with open(save_path + "train_config.json", "w") as f:
        json.dump(train_config, f)
    print("Train config saved:", save_path + "train_config.json")

    init_random_seed_torch(78)
    # gpus_list = [0, 1, 2, 4]
    # gpus_to_use(gpus_list)
    # device = get_device(gpus_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    print("Data preprocessing...")
    X_train, y_train = preprocess_data(data_train, tokenizer, punctuation_enc, segment_size)
    X_valid, y_valid = preprocess_data(data_valid, tokenizer, punctuation_enc, segment_size)

    print("Model initialization...")
    output_size = len(punctuation_enc)
    # bert_punc = nn.DataParallel(BertPunc(segment_size, output_size, train_config["model"]).to(device))
    bert_punc = BertPunc(segment_size, output_size, train_config["model"]).to(device)

    print("Top layer training...")
    data_loader_train = create_data_loader(X_train, y_train, batch_size=batch_size_top, shuffle=True)
    data_loader_valid = create_data_loader(X_valid, y_valid, batch_size=batch_size_top, shuffle=False)

    for p in bert_punc.bert.parameters():  # bert_punc.module.bert.parameters(): # nn.DataParallel
        p.requires_grad = False

    optimizer = optim.Adam(bert_punc.parameters(), lr=learning_rate_top)
    criterion = nn.CrossEntropyLoss()

    trainer_top = Trainer(
        device, bert_punc, optimizer, criterion, epochs_top, iterations_top,
        data_loader_train, data_loader_valid, punctuation_enc, save_path
    )
    best_val_loss = trainer_top.fit()

    print("All layers training...")
    # load the model to continue training
    bert_punc.load_state_dict(torch.load(trainer_top.best_model_path))
    data_loader_train = create_data_loader(X_train, y_train, batch_size=batch_size_all, shuffle=True)
    data_loader_valid = create_data_loader(X_valid, y_valid, batch_size=batch_size_all, shuffle=False)

    for p in bert_punc.bert.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(bert_punc.parameters(), lr=learning_rate_all)
    criterion = nn.CrossEntropyLoss()

    trainer_all = Trainer(
        device, bert_punc, optimizer, criterion, epochs_all, iterations_all,
        data_loader_train, data_loader_valid, punctuation_enc, save_path
    )

    best_val_loss = trainer_all.fit(best_val_loss=best_val_loss)
    print(f"\nbest_model_path: {trainer_all.best_model_path}, best_val_loss: {best_val_loss:.4f}")
