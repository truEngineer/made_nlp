import math
import time

import torch
from tqdm import tqdm

from train_utils import epoch_time


def train(model, device, loader, optimizer, criterion, clip, transformer=False):
    model.train()
    epoch_loss = 0

    for src, trg in tqdm(loader, total=len(loader), desc="train", position=0, leave=True):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        if not transformer:
            output = model(src, trg)  # trg: [trg_len, batch_size]
            # output: [trg_len, batch_size, output_dim]
            # As stated before, our decoder loop starts at 1, not 0.
            # This means the 0th element of our outputs tensor remains all zeros.
            # Here, when we calculate the loss, we cut off the first element of each tensor.
            # As the loss function only works on 2d inputs with 1d targets we need to flatten each of them with .view.
            output = output[1:].view(-1, output.shape[-1])  # output: [(trg_len - 1) * batch_size, output_dim]
            trg = trg[1:].view(-1)  # trg: [(trg_len - 1) * batch_size]
        else:
            # delete <eos> token trg[:,:-1]
            output, _ = model(src, trg[:, :-1])  # trg: [trg_len, batch_size]
            # output: [trg_len, batch_size, output_dim]
            output = output.contiguous().view(-1, output.shape[-1])
            # delete <sos> token trg[:,1:]
            trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # clip the gradients to prevent them from exploding (a common issue in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, device, loader, criterion, transformer=False):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in tqdm(loader, total=len(loader), desc="evaluate", position=0, leave=True):
            src, trg = src.to(device), trg.to(device)

            if not transformer:
                output = model(src, trg, 0)  # turn off teacher forcing
                # trg: [trg_len, batch_size], output: [trg_len, batch_size, output_dim]
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                # trg: [(trg_len - 1) * batch_size], output: [(trg_len - 1) * batch_size, output_dim]
            else:
                output, _ = model(src, trg[:, :-1])
                output = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_model(model, device, num_epochs,
                train_loader, val_loader,
                optimizer, criterion, scheduler, clip,
                model_name='baseline_model', transformer=False):
    train_history, val_history = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, device, train_loader, optimizer, criterion, clip, transformer)
        val_loss = evaluate(model, device, val_loader, criterion, transformer)
        scheduler.step(val_loss)

        end_time = time.time()

        train_history.append(train_loss)
        val_history.append(val_loss)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, f'{model_name}.pt')

    return train_history, val_history
