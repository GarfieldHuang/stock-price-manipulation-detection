"""
Preprocessing utilities for the training routines.
"""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from models.models import *
from hyperopt import STATUS_OK
from os.path import exists


class SequentialDataset(Dataset):
    """
    Class for sequential data sets.
    :param dataset_path: str or path to .parquet file.
    :param window: int, window size for timeseries.
    :param features: list of str's of columns to use as features.
    :param response: str of response column. None by default for unsupervised learning.
    """

    def __init__(self, dataset_path, window, features, response=None):
        self.data = pd.read_parquet(dataset_path, engine='fastparquet')
        self.window = window
        self.features = features

        if response is not None:
            self.response = [response]
            self.columns = self.features + self.response
        else:
            self.response = response
            self.columns = self.features

        # Deleting rows where at least a feature is missing
        self.data = self.data[~self.data[self.columns].isnull().any(1)]

        self.index_map = {}
        self.days = self.data.Date.unique()
        self.instruments_data_matrices = [[] for _ in range(len(self.days))]

        index = 0
        for day_index in range(len(self.days)):
            data_day = self.data[self.data.Date == self.days[day_index]]
            instruments_day_counts = data_day.groupby('ExternalSymbol').ExternalSymbol.count()
            instruments_day = instruments_day_counts[instruments_day_counts > self.window].index.tolist()
            for instrument_day_index in range(len(instruments_day)):
                data_day_instrument = data_day.loc[data_day.ExternalSymbol ==
                                                   instruments_day[instrument_day_index], self.columns].to_numpy()
                self.instruments_data_matrices[day_index].append(data_day_instrument)
                for book_instrument_day_index in range(data_day_instrument.shape[0] - self.window):
                    self.index_map[index] = (day_index, instrument_day_index, book_instrument_day_index)
                    index += 1
        del self.data

    def __getitem__(self, idx):
        day, instrument, book = self.index_map[idx]
        data = self.instruments_data_matrices[day][instrument][book: (book + self.window), :]

        if self.response:
            x = data[:, :-1]
            y = np.asarray(data[-1, -1])
            return torch.from_numpy(x).float(), torch.from_numpy(y).float()

        else:
            return torch.from_numpy(data).float(), torch.from_numpy(data).float()

    def __len__(self):
        return len(self.index_map)
  

def make_sequential_batches(dataset_path, window, feature_columns, response_column=None, shuffle=True, batch_size=512):
    """
    Create pytorch dataloaders from extracted limit order book datasets sequences.

    Parameters
    ----------
    dataset_path : str or Path
        The dataset for which to build dataloaders.
    window: int
        Size of window for time series
    feature_columns: list of str's
        The columns used as features.
    response_column: str
        The column used as response.
    shuffle: bool
        Shuffling of the batches.
    batch_size: int
        Batch size.
    Returns
    -------
    torch.data.DataLoader
        A PyTorch dataloader over the dataset returning (features, response) minibatches.
    """
    dataset = SequentialDataset(dataset_path, window, feature_columns, response_column)
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
    return batches


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


class Scheduler(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def model_train(model, task, epochs, train_batches, valid_batches, best):
    scheduler = None
    if task == 'seq-reconstruction-transformer':
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        scheduler = Scheduler(optimizer, dim_embed=model.d_model, warmup_steps=4000)
    else:
        optimizer = torch.optim.Adam(model.parameters())

    best_params = {'model': None, 'loss': np.float('inf')}
    best_epoch = 0
    checkpoint_epoch = 0

    if best['from_checkpoint']:
        if exists(r'models/checkpoint_' + type(model).__name__ + '.pt'):
            checkpoint = torch.load(r'models/checkpoint_' + type(model).__name__ + '.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_epoch = checkpoint['epoch']
            best_epoch = checkpoint_epoch
            best_params['loss'] = checkpoint['loss']

    for epoch in range(checkpoint_epoch+1, epochs+1):
        print(f"Epoch {epoch}")
        model.train()
        train_loss_ = 0.

        for count, (features, responses) in enumerate(train_batches):
            if torch.cuda.is_available():
                features, responses = features.cuda(), responses.cuda()

            if task == 'seq-reconstruction':
                predictions = model(features)
                train_loss = (predictions - responses).pow(2).sum(1).sum(1).mean()
            elif task == 'seq-reconstruction-transformer':
                predictions = model(features, mode="train")
                train_loss = (predictions - responses).pow(2).sum(1).sum(1).mean()
            else:
                raise Exception(f"Unknown task '{task}'")

            train_loss_ += train_loss.item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if count % 500 == 0:
                print(f"Batch {count: >6}/{len(train_batches)}, train loss {train_loss.cpu():.2f}")

        train_loss_ /= len(train_batches)
        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for count, (features, responses) in enumerate(valid_batches):
                if torch.cuda.is_available():
                    features, responses = features.cuda(), responses.cuda()

                if task == 'seq-reconstruction':
                    predictions = model(features)
                    valid_loss += (predictions - responses).pow(2).sum(1).sum(1).mean().item()
                elif task == 'seq-reconstruction-transformer':
                    predictions = model(features, mode="generate")
                    valid_loss += (predictions - responses).pow(2).sum(1).sum(1).mean().item()

            valid_loss /= len(valid_batches)

        if best_params['loss'] / valid_loss > 1.05:
            best_epoch = epoch
        if valid_loss < best_params['loss']:
            best_params['model'] = model
            best_params['loss'] = valid_loss
            if valid_loss < best['loss']:
                torch.save(model, 'models/best_' + type(model).__name__ + '.pt')
                best['loss'] = valid_loss
                best['model'] = model

        if epoch - best_epoch > 100:
            break

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_params['loss'],
                    'nb_iter': best['nb_iter'],
                    'best_loss': best['loss']},
                    'models/checkpoint_' + type(best_params['model']).__name__ + '.pt')

        print(f"Train loss: {train_loss_:.4f}, Validation loss {valid_loss:.4f}\n")

    return best_params


def optimization(params, task, model, best, epochs, train_batches, valid_batches):
    if model == 'transformer-autoencoder':
        nb_features = params['nb_features']
        d_model = params['d_model']
        dim_ff = params['dim_ff']
        seq_length = params['seq_length']
        num_layers = params['num_layers']
        if not best['from_checkpoint']:
            if torch.cuda.is_available():
                model = TransformerAutoencoder(nb_features, seq_length, d_model, dim_ff, num_layers).cuda()
            else:
                model = TransformerAutoencoder(nb_features, seq_length, d_model, dim_ff, num_layers)
        else:
            if torch.cuda.is_available():
                model = torch.load('models/best_TransformerAutoencoder.pt').cuda()
            else:
                model = torch.load('models/best_TransformerAutoencoder.pt')

    elif model == 'stackedlstm-autoencoder':
        nb_features = params['nb_features']
        hidden_units = params['hidden_units']
        layers = params['layers']
        if not best['from_checkpoint']:
            if torch.cuda.is_available():
                model = StackedLSTMAutoencoder(nb_features, hidden_units, layers).cuda()
            else:
                model = StackedLSTMAutoencoder(nb_features, hidden_units, layers)
        else:
            if torch.cuda.is_available():
                model = torch.load('models/best_StackedLSTMAutoencoder.pt').cuda()
            else:
                model = torch.load('models/best_StackedLSTMAutoencoder.pt')

    elif model == 'mlp-autoencoder':
        hidden_units = params['hidden_units']
        seq_length = params['seq_length']
        latent_dim = params['latent_dim']
        nb_layers = params['nb_layers']
        activation = params['activation']
        nb_features = params['nb_features']
        if not best['from_checkpoint']:
            if torch.cuda.is_available():
                model = MLPAutoencoder(nb_features, seq_length, latent_dim, nb_layers, hidden_units, activation).cuda()
            else:
                model = MLPAutoencoder(nb_features, seq_length, latent_dim, nb_layers, hidden_units, activation)
        else:
            if torch.cuda.is_available():
                model = torch.load('models/best_MLPAutoencoder.pt').cuda()
            else:
                model = torch.load('models/best_MLPAutoencoder.pt')

    else:
        print('Model not implemented.')

    best_params = model_train(model, task, epochs, train_batches, valid_batches, best)

    best['nb_iter'] += 1
    best['from_checkpoint'] = False

    log = 'Trial {}: MSE = {} --- Running Best MSE = {}.'.format(best['nb_iter'], round(best_params['loss'], 2),
                                                                 round(best['loss'], 2))
    print(log)
    return {'loss': best_params['loss'], 'status': STATUS_OK}

