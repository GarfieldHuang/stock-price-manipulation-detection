"""
Preprocess the LOBSTER data. Raw data should be kept in the data folder, and the resulting
generated data files ready for passing to a machine learning model will be stored directly in data/.
"""

import pandas as pd
import numpy as np
import os
import argparse
from multiprocessing import Pool
from preprocessing.utilities import data_norm_and_features, split_data
import warnings
warnings.filterwarnings("ignore")


def main(data_folder, inter_save_folder, prep_save_folder, save):
    files = os.listdir(data_folder)
    dates = list(set([os.path.join(data_folder, f)[-12:-4] for f in files if 'MarketDepth' in f]))
    dates.sort()

    if not os.path.exists(inter_save_folder):
        os.makedirs(inter_save_folder)

    # Remove any old files in the intermediate directory
    for file in os.listdir(inter_save_folder):
        if 'parquet' or 'csv' in file:
            os.remove(os.path.join(inter_save_folder, file))

    if not os.path.exists(prep_save_folder):
        os.makedirs(prep_save_folder)
    
    # Process each day independently
    for date in dates:
        print('Preprocessing ' + str(date) + ':')
        market_data_path = data_folder + '/MarketDepth_' + str(date) + '.csv'
        market_data = pd.read_csv(market_data_path, delimiter=";")
        daily_stocks = market_data.ExternalSymbol.str[0:3].unique()
        daily_stocks.sort()

        for stock in daily_stocks:
            print(stock)
            destination_file = inter_save_folder + '/NormalizedFeatures_' + stock + "_" + str(date) + '.parquet'
            destination_file_features = inter_save_folder + '/Features_' + stock + "_" + str(date) + '.csv'

            # Importing data for options with that underlying stock
            stock_data = market_data[market_data.ExternalSymbol.str[0:3] == stock]

            # Computing features
            print('     1. Computing features.')
            if save:
                stock_data, feature_data = data_norm_and_features(stock_data, save)
                stock_data = stock_data.sort_values(by=['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']
                                                    ).sort_index().reset_index(drop=True)
                feature_data = feature_data.sort_values(by=['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']
                                                        ).sort_index().reset_index(drop=True)

            else:
                stock_data = data_norm_and_features(stock_data, save)
                stock_data = stock_data.sort_values(by=['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber'],
                                                    ).sort_index().reset_index(drop=True)

            print('     2. Saving.')
            stock_data['OriginalSequenceNumber'] = stock_data['OriginalSequenceNumber'].astype(str)
            stock_data.columns = [str(column) for column in stock_data.columns]
            stock_data.to_parquet(destination_file, index=False, engine='fastparquet')
            del stock_data

            if save:
                feature_data.to_csv(destination_file_features, index=False)
                del feature_data

        print("\n")

    print('Aggregating daily files.')

    # Aggregating and saving feature files.
    if save:
        total_features_df = pd.DataFrame()

        prep_feature_files = [f for f in os.listdir(inter_save_folder) if f.endswith('.' + 'csv')]

        for prep_file in prep_feature_files:
            with open(os.path.join(inter_save_folder, prep_file), 'rb') as f:
                df = pd.read_csv(f)
            total_features_df = total_features_df.append(df, ignore_index=True)
        total_features_df = total_features_df.sort_values(by=['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber'])
        del df

        total_features_df.to_csv(inter_save_folder + '/Features_Total.csv', index=False)
        del total_features_df

    # Aggregating normalized data.
    prep_files = [file for file in os.listdir(inter_save_folder) if '.parquet' in file]

    dfs = []
    for prep_file in prep_files:
        with open(os.path.join(inter_save_folder, prep_file).replace("\\", "/"), 'rb') as f:
            df = pd.read_parquet(f, engine='fastparquet')
        dfs.append(df)
    total_df = pd.concat(dfs).sort_values(by=['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber'])

    print('Saving aggregated data.')
    total_df.to_parquet(inter_save_folder + '/NormalizedFeatures_Total.parquet', index=False,
                        engine='fastparquet', row_group_offsets=20000)


def split_train_test(inter_save_folder, prep_save_folder):
    print('Splitting aggregated data into train, valid and test sets.')

    # Getting all the data
    total_df = pd.read_parquet(inter_save_folder + '/NormalizedFeatures_Total.parquet', engine='fastparquet')
    anomalies = pd.read_csv('data/formatted/Anomalies_formatted.csv', sep=';')
    anomalies['Fraud'] = 1
    anomalies['OriginalSequenceNumber'] = anomalies['OriginalSequenceNumber'].astype(str)

    # Merging anomalies and normal data into a single data frame
    total_df = total_df.merge(anomalies.drop(['FraudType'], axis=1), how='left',
                              on=['Date', 'Time', 'ExternalSymbol', 'OriginalSequenceNumber'])
    total_df = total_df.drop(['MilliSeconds_y'], axis=1)
    total_df = total_df.rename(columns={'MilliSeconds_x': 'MilliSeconds'})
    total_df['Fraud'] = total_df['Fraud'].fillna(value=0)

    # Splitting data into train/valid/test
    split_data_sets = split_data(total_df, anomalies)
    train_data = split_data_sets['train'].sort_values(['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']
                                                      ).reset_index(drop=True)
    valid_data = split_data_sets['valid'].sort_values(['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']
                                                      ).reset_index(drop=True)
    test_data = split_data_sets['test'].sort_values(['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']
                                                    ).reset_index(drop=True)

    # Oversampling less frequent instruments for equal representation in train and valid sets
    # On train set
    counts = train_data['ExternalSymbol'].value_counts()
    instruments = counts.index.values
    max_rows = counts.values[np.argsort(counts.values)[-2]]

    for i, instrument in enumerate(instruments):
        repeats = max_rows // counts.values[i]
        instrument_data = train_data[train_data['ExternalSymbol'] == instrument]

        if repeats > 1:
            for repeat in range(repeats-1):
                new_data = instrument_data.copy()
                new_data['Date'] = (new_data['Date'].str[:4].astype(int) + repeat).astype(str) + new_data['Date'].str[4:]
                train_data = pd.concat((train_data, new_data)).reset_index(drop=True)

    # On valid set
    counts = valid_data['ExternalSymbol'].value_counts()
    instruments = counts.index.values
    max_rows = counts.values[np.argsort(counts.values)[-2]]

    for i, instrument in enumerate(instruments):
        repeats = max_rows // counts.values[i]
        instrument_data = valid_data[valid_data['ExternalSymbol'] == instrument]

        if repeats > 1:
            for repeat in range(repeats-1):
                new_data = instrument_data.copy()
                new_data['Date'] = (new_data['Date'].str[:4].astype(int) + repeat).astype(str) + new_data['Date'].str[4:]
                valid_data = pd.concat((valid_data, new_data)).reset_index(drop=True)

    train_data['FraudType'] = train_data['FraudType'].fillna(0)
    valid_data['FraudType'] = valid_data['FraudType'].fillna(0)
    test_data['FraudType'] = test_data['FraudType'].fillna(0)

    train_data['index'] = train_data.index.values
    valid_data['index'] = valid_data.index.values
    test_data['index'] = test_data.index.values

    print('Saving train, valid and test sets.')
    train_data.to_parquet(prep_save_folder + '/train_data.parquet', index=False, engine='fastparquet',
                          row_group_offsets=100000)
    valid_data.to_parquet(prep_save_folder + '/valid_data.parquet', index=False, engine='fastparquet',
                          row_group_offsets=100000)
    test_data.to_parquet(prep_save_folder + '/test_data.parquet', index=False, engine='fastparquet',
                         row_group_offsets=100000)

    print('Preprocessing done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--save',
        help='bool, save features before normalization (default True)',
        type=bool,
        default=False
    )

    args = parser.parse_args()
    save = args.save

    main('data/formatted/', 'data/intermediate', 'data/preprocessed', save)
    split_train_test('data/intermediate', 'data/preprocessed')

    
    

