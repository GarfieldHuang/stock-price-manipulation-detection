"""
Train the models.
"""
from train.utilities import make_sequential_batches, optimization
import numpy as np
import torch
from hyperopt import fmin, tpe, hp, Trials
from functools import partial
import warnings
from os.path import exists
warnings.filterwarnings("ignore")

feature_columns_ae = ['ReturnBid1', 'ReturnAsk1', 'DerivativeReturnBid1', 'DerivativeReturnAsk1', 'BidSize1',
                      'AskSize1', 'TradeBidSize', 'TradeAskSize', 'CancelledBidSize', 'CancelledAskSize',
                      'TradeBidIndicator', 'TradeAskIndicator', 'CancelledBidIndicator', 'CancelledAskIndicator']


def transformer_autoencoder(seq_length, restart_from_checkpoint):
    train_batches = make_sequential_batches('data/preprocessed_lobster/train_data.parquet',
                                            seq_length, feature_columns_ae)
    valid_batches = make_sequential_batches('data/preprocessed_lobster/valid_data.parquet',
                                            seq_length, feature_columns_ae)

    space = {'d_model': 128,
             'nb_features': len(feature_columns_ae),
             'seq_length': seq_length,
             'dim_ff': 512,
             'num_layers': 6}

    best = {'model': None, 'loss': np.float('inf'), 'nb_iter': 0, 'from_checkpoint': False}

    if restart_from_checkpoint:
        if exists(r'models/checkpoint_TransformerAutoencoder.pt'):
            checkpoint = torch.load(r'models/checkpoint_TransformerAutoencoder.pt')
            best['loss'] = checkpoint['best_loss']
            best['nb_iter'] = checkpoint['nb_iter']
            best['from_checkpoint'] = True
        else:
            print('No checkpoint available for this model. Starting new training schedule.')

    # Transformer encoder training by sequence input reconstruction
    trials_pretrain = Trials()
    objective = partial(optimization, task='seq-reconstruction-transformer', model='transformer-autoencoder',
                        best=best, epochs=500, train_batches=train_batches, valid_batches=valid_batches)
    fmin(objective, space, algo=tpe.suggest, max_evals=10-best['nb_iter'], trials=trials_pretrain)


def stackedlstm_autoencoder(seq_length, restart_from_checkpoint):
    train_batches = make_sequential_batches('data/preprocessed_lobster/train_data.parquet',
                                            seq_length, feature_columns_ae)
    valid_batches = make_sequential_batches('data/preprocessed_lobster/valid_data.parquet',
                                            seq_length, feature_columns_ae)

    space = {'hidden_units': 128,
             'layers': 6,
             'nb_features': len(feature_columns_ae)}

    best = {'model': None, 'loss': np.float('inf'), 'nb_iter': 0, 'from_checkpoint': False}

    if restart_from_checkpoint:
        checkpoint = torch.load(r"models/checkpoint_StackedLSTMAutoencoder.pt")
        best['loss'] = checkpoint['loss']
        best['nb_iter'] = checkpoint['nb_iter']
        best['from_checkpoint'] = True

    trials = Trials()
    objective = partial(optimization, task='seq-reconstruction', model='stackedlstm-autoencoder',
                        best=best, epochs=500, train_batches=train_batches, valid_batches=valid_batches)
    fmin(objective, space, algo=tpe.suggest, max_evals=10-best['nb_iter'], trials=trials)


def mlp_autoencoder(seq_length, restart_from_checkpoint):
    train_batches = make_sequential_batches('data/preprocessed_lobster/train_data.parquet',
                                            seq_length, feature_columns_ae)
    valid_batches = make_sequential_batches('data/preprocessed_lobster/valid_data.parquet',
                                            seq_length, feature_columns_ae)

    space = {'hidden_units': 256,
             'seq_length': seq_length,
             'latent_dim': 128,
             'nb_layers': 6,
             'activation': 'relu',
             'nb_features': len(feature_columns_ae)}

    best = {'model': None, 'loss': np.float('inf'), 'nb_iter': 0, 'from_checkpoint': False}

    if restart_from_checkpoint:
        checkpoint = torch.load(r"models/checkpoint_MLPAutoencoder.pt")
        best['loss'] = checkpoint['loss']
        best['nb_iter'] = checkpoint['nb_iter']
        best['from_checkpoint'] = True

    trials = Trials()
    objective = partial(optimization, task='seq-reconstruction', model='mlp-autoencoder',
                        best=best, epochs=500, train_batches=train_batches, valid_batches=valid_batches)
    fmin(objective, space, algo=tpe.suggest, max_evals=10-best['nb_iter'], trials=trials)


if __name__ == '__main__':
    seq_length = 25  # sequence length of the model

    transformer_autoencoder(seq_length, False)
    # stackedlstm_autoencoder(seq_length, False)
    # mlp_autoencoder(seq_length, False)