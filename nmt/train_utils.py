import random

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def init_weights_xavier(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def remove_special_tokens(text, specials=('<unk>', '<pad>', '<bos>', '<eos>')):
    return [t for t in text if t not in specials]


def get_text(x, trg_vocab):
    # get_itos() -> List[str], Returns: List mapping indices to tokens.
    idx2token = trg_vocab.get_itos()
    text = [idx2token[idx] for idx in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_special_tokens(text)  # remove '<unk>', '<pad>' tokens
    if len(text) < 1:
        text = []
    return text


# transformer
def translate_sentence_vectorized(src_tensor, trg_vocab, model, device, max_len=100):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)  # enc_src: [batch_sz, src_len, hid_dim]

    trg_indexes = [[trg_vocab['<bos>']] for _ in range(len(src_tensor))]  # trg_vocab.stoi[trg_vocab.init_token]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_tokens = output.argmax(2)[:, -1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_vocab['<eos>']:  # trg_field.vocab.stoi[trg_field.eos_token]
                # PAD_IDX = en_vocab['<pad>'] # PAD_IDX = en_vocab.get_stoi()['<pad>']
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_vocab['<eos>']:  # trg_field.vocab.stoi[trg_field.eos_token]
                break
            pred_sentence.append(trg_vocab.get_itos()[trg_sentence[i]])  # trg_field.vocab.itos[trg_sentence[i]]
        pred_sentence = remove_special_tokens(pred_sentence)
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def bleu_score(model, device, test_loader, src_vocab, trg_vocab, transformer=False, src_bert_tokenizer=None):
    model.eval()
    input_text, target_text, generated_text = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="bleu", position=0, leave=True):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            if transformer:
                output, _ = translate_sentence_vectorized(src, trg_vocab, model, device)  # output: List
                if src_bert_tokenizer is not None:  # BERT + Transformer encoder
                    input_text.extend(
                        [src_bert_tokenizer.decode(x, skip_special_tokens=True).split() for x in src.cpu().numpy()]
                    )
                else:  # Transformer encoder
                    input_text.extend([get_text(x, src_vocab) for x in src.cpu().numpy()])
                target_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy()])
                generated_text.extend(output)
            else:  # RNN Seq2Seq
                output = model(src, trg, 0)  # turn off teacher forcing
                output = output.argmax(dim=-1).cpu()  # torch.Size([43, 128, 10799]) -> torch.Size([43, 128])
                if src_bert_tokenizer is not None:  # BERT + RNN encoder
                    input_text.extend(
                        [src_bert_tokenizer.decode(x, skip_special_tokens=True).split() for x in src.cpu().numpy().T]
                    )
                else:  # Embeddings encoder
                    input_text.extend([get_text(x, src_vocab) for x in src.cpu().numpy().T])
                target_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
                generated_text.extend(
                    [get_text(x, trg_vocab) for x in output.cpu().numpy().T])  # torch.Size([128, 42])

    score = corpus_bleu([[text] for text in target_text], generated_text) * 100
    return score, input_text, target_text, generated_text


def calc_bleu(model, device, test_loader, src_vocab, trg_vocab,
              transformer=False, src_bert_tokenizer=None, num_examples=5):
    score, input_text, target_text, generated_text = bleu_score(
        model, device, test_loader, src_vocab, trg_vocab,
        transformer=transformer, src_bert_tokenizer=src_bert_tokenizer,
    )
    print(f'bleu: {score:.3f}\n')
    for _ in range(num_examples):
        index = random.randint(0, len(target_text))
        print('input:', " ".join(input_text[index]))
        print('target:', " ".join(target_text[index]))
        print('generated:', " ".join(generated_text[index]), '\n')


def plot_history(train_history, val_history):
    with plt.style.context('seaborn'):
        plt.figure(figsize=(4, 2))
        plt.plot(range(1, len(train_history) + 1), train_history, label='train')
        plt.plot(range(1, len(val_history) + 1), val_history, label='val')
        plt.legend()
        plt.show()
