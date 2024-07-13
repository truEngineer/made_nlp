import os
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self, device: torch.device, model: nn.Module,
            optimizer: optim.Optimizer, criterion: nn.modules.loss,
            epochs: int, iterations: int,
            data_loader_train: DataLoader, data_loader_valid: DataLoader,
            punctuation_enc: dict, save_path: str,
    ):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.iterations = iterations
        self.data_loader_train = data_loader_train
        self.data_loader_valid = data_loader_valid
        self.punctuation_enc = punctuation_enc
        self.save_path = save_path
        self.best_model_path = None

    def _validate(
        self, epoch: int, iteration: int, train_loss: float, best_val_loss: float
    ) -> float:
        val_losses, val_accs, val_f1s = [], [], []
        label_keys = list(self.punctuation_enc.keys())
        label_vals = list(self.punctuation_enc.values())

        for inputs, labels in tqdm(self.data_loader_valid, total=len(self.data_loader_valid)):
            with torch.no_grad():
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                val_loss = self.criterion(output, labels)
                val_losses.append(val_loss.cpu().data.numpy())
                y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()
                y_true = labels.cpu().data.numpy().flatten()
                val_accs.append(metrics.accuracy_score(y_true, y_pred))
                val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        val_f1 = np.array(val_f1s).mean(axis=0)

        improved = ''
        # model_path = self.save_path + f'model_{epoch:02d}_{iteration:02d}'
        # torch.save(self.model.state_dict(), model_path)
        if val_loss < best_val_loss:
            improved = '*'
            best_val_loss = val_loss
            # model_path = self.save_path + f'model_{epoch:02d}_{iteration:02d}.pt'
            model_path = self.save_path + 'model.pt'
            torch.save(self.model.state_dict(), model_path)
            self.best_model_path = model_path

        f1_cols = ';'.join(['f1_' + key for key in label_keys])
        progress_path = self.save_path + 'train_log.csv'
        if not os.path.isfile(progress_path):
            with open(progress_path, 'w') as f:
                f.write('time;epoch;iteration;train loss;val loss;accuracy;' + f1_cols + '\n')

        f1_vals = ';'.join([f'{val:.4f}' for val in val_f1])
        with open(progress_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')};{epoch + 1};{iteration};" +
                    f"{train_loss:.4f};{val_loss:.4f};{val_acc:.4f};{f1_vals}\n")

        print(f"Epoch: {epoch + 1}/{self.epochs}",
              f"Iteration: {iteration}/{self.iterations}",
              f"Train Loss: {train_loss:.4f}", f"Val Loss: {val_loss:.4f}",
              f"Accuracy: {val_acc:.4f}", f"F1: {f1_vals}", improved)

        return best_val_loss

    def _train(self, best_val_loss: float) -> float:
        print_every = len(self.data_loader_train) // self.iterations + 1
        self.model.train()
        pbar = tqdm(total=print_every)

        train_loss = np.inf
        for e in range(self.epochs):
            counter = 1
            iteration = 1

            for inputs, labels in self.data_loader_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss = loss.cpu().data.numpy()

                pbar.update()

                if counter % print_every == 0:
                    pbar.close()
                    self.model.eval()
                    best_val_loss = self._validate(
                        e, iteration, train_loss, best_val_loss
                    )
                    self.model.train()
                    pbar = tqdm(total=print_every)
                    iteration += 1

                counter += 1

            pbar.close()
            self.model.eval()
            best_val_loss = self._validate(
                e, iteration, train_loss, best_val_loss
            )
            self.model.train()
            if e < self.epochs - 1:
                pbar = tqdm(total=print_every)

        return best_val_loss  # return model, optimizer, best_val_loss

    def fit(self, best_val_loss: float = np.inf) -> float:
        best_val_loss = self._train(best_val_loss)
        # Load best model -> inference mode
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        return best_val_loss
