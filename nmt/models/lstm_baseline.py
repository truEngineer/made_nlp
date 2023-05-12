import random

import torch
import torch.nn as nn

# PyTorch Seq2Seq https://github.com/bentrevett/pytorch-seq2seq
# Sequence to Sequence Learning with Neural Networks.ipynb
# https://habr.com/ru/articles/567142/


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=hid_dim,
            num_layers=num_layers, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        embedded = self.embedding(src)  # src: [src_len, batch_size]
        embedded = self.dropout(embedded)  # embedded: [src_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [src_len, batch size, hid dim * n directions]
        # hidden: [num_layers * num_directions, batch_size, hid_dim]
        # cell: [num_layers * num_directions, batch_size, hid_dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=hid_dim,
            num_layers=num_layers, dropout=dropout
        )
        self.out = nn.Linear(in_features=hid_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [num_layers * num_directions, batch_size, hid_dim]
        # cell: [num_layers * num_directions, batch_size, hid_dim]

        # num_directions in the decoder will both always be 1, therefore:
        # hidden: [n layers, batch size, hid dim]
        # context: [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # input: [batch_size] -> [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # embedded: [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [seq_len, batch_size, hid_dim * n directions]
        # hidden: [num_layers * num_directions, batch_size, hid_dim]
        # cell: [num_layers * num_directions, batch_size, hid_dim]
        # As we are only decoding one token at a time, the input tokens will always have a sequence length of 1.
        # seq_len and num_directions will always be 1 in the decoder, therefore:
        # output: [1, batch_size, hid_dim]
        # hidden: [num_layers, batch_size, hid_dim]
        # cell: [num_layers, batch_size, hid_dim]
        prediction = self.out(output.squeeze(0))  # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        input = trg[0, :]  # first input to the decoder is the <sos> tokens
        # Our decoder loop starts at 1, not 0. This means the 0th element of our outputs tensor remains all zeros.
        # Later on when we calculate the loss, we cut off the first element of each tensor.
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)  # output: [batch_size, output_dim=len(target_vocab)]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)  # top1 = output.max(1)[1]
            # if teacher forcing, use actual next token as next input if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs
