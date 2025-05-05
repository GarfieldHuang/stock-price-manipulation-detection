import numpy as np
import pandas as pd
import random


def data_norm_and_features(data, save_features=False):
    """
    Normalizes the LOB and constructs predictive features based on the LOB at each time step.
    :param data: formated LOB data in dataframe format.
    :param save_features: bool, save the non-normalized features in a .csv file
    :return: Normalized features.
    """
    data = data[data['GroupStatus'] == "Continuous trading"]

    external_symbols = data.ExternalSymbol.unique()

    for symbol in external_symbols:
        symbol_data = data[data['ExternalSymbol'] == symbol]
        indexes = symbol_data.index.values

        # Computing delta time between two LOB updates
        time_diff = symbol_data['TimeInMilliSecs'].diff().values * \
                    (symbol_data['TimeInMilliSecs'].diff().values >= 1).astype(int)
        time_diff[np.isnan(time_diff)] = 1
        time_diff[time_diff == 0] = 1

        symbol_data['TradeBidIndicator'] = (symbol_data['TradeIndicator'] == 1).astype(int)
        symbol_data['TradeAskIndicator'] = (symbol_data['TradeIndicator'] == -1).astype(int)

        # Computing absolute price move and derivative of move in time
        symbol_data['ReturnBid1'] = np.log(symbol_data['BidPrice1'] / symbol_data['BidPrice1'].shift(-1))
        symbol_data['ReturnAsk1'] = np.log(symbol_data['AskPrice1'] / symbol_data['AskPrice1'].shift(-1))

        symbol_data['DerivativeReturnBid1'] = (symbol_data['ReturnBid1'] / time_diff)
        symbol_data['DerivativeReturnAsk1'] = (symbol_data['ReturnAsk1'] / time_diff)

        # Inserting features back into main dataframe
        data.loc[indexes, 'ReturnBid1'] = symbol_data['ReturnBid1']
        data.loc[indexes, 'ReturnAsk1'] = symbol_data['ReturnAsk1']
        data.loc[indexes, 'DerivativeReturnBid1'] = symbol_data['DerivativeReturnBid1']
        data.loc[indexes, 'DerivativeReturnAsk1'] = symbol_data['DerivativeReturnAsk1']
        data.loc[indexes, 'BidSize1'] = symbol_data['BidSize1'].rolling(window=10).mean()
        data.loc[indexes, 'AskSize1'] = symbol_data['AskSize1'].rolling(window=10).mean()
        data.loc[indexes, 'TradeBidSize'] = symbol_data['TradeSize'].abs().rolling(window=10).mean() * (symbol_data['TradeBidIndicator'] == 1).astype(int)
        data.loc[indexes, 'TradeAskSize'] = symbol_data['TradeSize'].abs().rolling(window=10).mean() * (symbol_data['TradeAskIndicator'] == 1).astype(int)
        data.loc[indexes, 'CancelledBidSize'] = symbol_data['CancelledBidSize'].rolling(window=10).mean()
        data.loc[indexes, 'CancelledAskSize'] = symbol_data['CancelledAskSize'].rolling(window=10).mean()
        data.loc[indexes, 'TradeBidIndicator'] = symbol_data['TradeBidIndicator'] / time_diff
        data.loc[indexes, 'TradeAskIndicator'] = symbol_data['TradeAskIndicator'] / time_diff
        data.loc[indexes, 'CancelledBidIndicator'] = symbol_data['CancelledBidIndicator'] / time_diff
        data.loc[indexes, 'CancelledAskIndicator'] = symbol_data['CancelledAskIndicator'] / time_diff

    # Saving the features before normalization
    features_names = ['Date', 'Time', 'MilliSeconds', 'TimeInMilliSecs', 'ExternalSymbol', 'OriginalSequenceNumber',
                      'FraudType', 'ReturnBid1', 'ReturnAsk1', 'DerivativeReturnBid1', 'DerivativeReturnAsk1',
                      'BidSize1', 'AskSize1', 'TradeBidSize', 'TradeAskSize', 'CancelledBidSize', 'CancelledAskSize',
                      'TradeBidIndicator', 'TradeAskIndicator', 'CancelledBidIndicator', 'CancelledAskIndicator']

    if save_features:
        feature_data = data[features_names].copy()

    # Normalization of the features
    for symbol in external_symbols:
        indexes = data[data['ExternalSymbol'] == symbol].index.values
        fraud_free_indexes = data[(data['ExternalSymbol'] == symbol) & (pd.isna(data['FraudType']))].index.values

        # Normalize data on daily instrument basis
        data.loc[indexes, 'TradeBidSize'] = (data.loc[indexes, 'TradeBidSize'] - data.loc[fraud_free_indexes, 'TradeBidSize'].mean()) / data.loc[fraud_free_indexes, 'TradeBidSize'].std()
        data.loc[indexes, 'TradeAskSize'] = (data.loc[indexes, 'TradeAskSize'] - data.loc[fraud_free_indexes, 'TradeAskSize'].mean()) / data.loc[fraud_free_indexes, 'TradeAskSize'].std()
        data.loc[indexes, 'CancelledBidSize'] = (data.loc[indexes, 'CancelledBidSize'] - data.loc[fraud_free_indexes, 'CancelledBidSize'].mean()) / data.loc[fraud_free_indexes, 'CancelledBidSize'].std()
        data.loc[indexes, 'CancelledAskSize'] = (data.loc[indexes, 'CancelledAskSize'] - data.loc[fraud_free_indexes, 'CancelledAskSize'].mean()) / data.loc[fraud_free_indexes, 'CancelledAskSize'].std()
        data.loc[indexes, 'TradeBidIndicator'] = (data.loc[indexes, 'TradeBidIndicator'] - data.loc[fraud_free_indexes, 'TradeBidIndicator'].mean()) / data.loc[fraud_free_indexes, 'TradeBidIndicator'].std()
        data.loc[indexes, 'TradeAskIndicator'] = (data.loc[indexes, 'TradeAskIndicator'] - data.loc[fraud_free_indexes, 'TradeAskIndicator'].mean()) / data.loc[fraud_free_indexes, 'TradeAskIndicator'].std()
        data.loc[indexes, 'CancelledBidIndicator'] = (data.loc[indexes, 'CancelledBidIndicator'] - data.loc[fraud_free_indexes, 'CancelledBidIndicator'].mean()) / data.loc[fraud_free_indexes, 'CancelledBidIndicator'].std()
        data.loc[indexes, 'CancelledAskIndicator'] = (data.loc[indexes, 'CancelledAskIndicator'] - data.loc[fraud_free_indexes, 'CancelledAskIndicator'].mean()) / data.loc[fraud_free_indexes, 'CancelledAskIndicator'].std()
        data.loc[indexes, 'ReturnBid1'] = (data.loc[indexes, 'ReturnBid1'] - data.loc[fraud_free_indexes, 'ReturnBid1'].mean()) / data.loc[fraud_free_indexes, 'ReturnBid1'].std()
        data.loc[indexes, 'ReturnAsk1'] = (data.loc[indexes, 'ReturnAsk1'] - data.loc[fraud_free_indexes, 'ReturnAsk1'].mean()) / data.loc[fraud_free_indexes, 'ReturnAsk1'].std()
        data.loc[indexes, 'DerivativeReturnBid1'] = (data.loc[indexes, 'DerivativeReturnBid1'] - data.loc[fraud_free_indexes, 'DerivativeReturnBid1'].mean()) / data.loc[fraud_free_indexes, 'DerivativeReturnBid1'].std()
        data.loc[indexes, 'DerivativeReturnAsk1'] = (data.loc[indexes, 'DerivativeReturnAsk1'] - data.loc[fraud_free_indexes, 'DerivativeReturnAsk1'].mean()) / data.loc[fraud_free_indexes, 'DerivativeReturnAsk1'].std()
        data.loc[indexes, 'BidSize1'] = (data.loc[indexes, 'BidSize1'] - data.loc[fraud_free_indexes, 'BidSize1'].mean()) / data.loc[fraud_free_indexes, 'BidSize1'].std()
        data.loc[indexes, 'AskSize1'] = (data.loc[indexes, 'AskSize1'] - data.loc[fraud_free_indexes, 'AskSize1'].mean()) / data.loc[fraud_free_indexes, 'AskSize1'].std()

    if save_features:
        return data[features_names], feature_data
    else:
        return data[features_names]


def split_data(data, anomalies):
    """
    Splits a limit order book file into train/valid/test

    Parameters
    ----------
    data : pandas Dataframe
        Dataframe of limit order book data
    anomalies : pandas Dataframe
        Dataframe of the location of the anomalies in the LOB file.
        
    Returns
    -------
    Dictionary of train_data, valid_data and test_data with the sequential anomalies.
    """
    time = pd.to_datetime(anomalies.Time, format='%H%M%S')
    anomalies['Hours'] = time.dt.hour
    anomalies['Minutes'] = time.dt.minute
    anomalies['Seconds'] = time.dt.second
    anomalies['TimeInMilliSecs'] = (anomalies.Hours * 60 * 60 * 1000 + anomalies.Minutes * 60 * 1000 +
                                    anomalies.Seconds * 1000 + anomalies.MilliSeconds)
    normal_data, anomalous_data = [], []
    sequence_idx = 0

    # Splitting the data on a per-symbol, per-day basis
    for symbol in data['ExternalSymbol'].unique():
        # Getting data for a single stock/symbol
        symbol_data = data[data['ExternalSymbol'] == symbol]
        symbol_anomalies = anomalies[anomalies['ExternalSymbol'] == symbol]

        for date in symbol_data['Date'].unique():
            # Getting single day of the stock data
            daily_symbol_data = symbol_data[symbol_data['Date'] == date]
            daily_symbol_anomalies = symbol_anomalies[symbol_anomalies['Date'] == date].sort_values(['TimeInMilliSecs', 'OriginalSequenceNumber'])

            # Getting all the fraud clusters' numbers
            clusters = daily_symbol_anomalies['ClusterNo'].unique()
            nb_clusters = len(clusters)

            # If there are frauds in the data sample, then splitting the data into before fraud/fraud/after fraud sets
            # for each fraud/cluster. Before fraud and after fraud are fraud-free sets used for training and validation,
            # whereas the fraud and surrounding LOB events are used for testing.
            if nb_clusters != 0:
                anomalies_starts = []
                anomalies_stops = []

                if nb_clusters > 1:
                    # Getting the anomaly start 35 seconds before the actual first fraudulent order, to create a test
                    # set containing mostly normal orders. Also setting anomaly stop 35 seconds after last fraudulent
                    # order. Do this for all fraud clusters.
                    anomalies_starts.append(daily_symbol_anomalies['TimeInMilliSecs'].values[0] - 35 * 1000)

                    for i, cluster in enumerate(clusters[1:]):
                        anomaly_start = daily_symbol_anomalies[daily_symbol_anomalies['ClusterNo'] == cluster]['TimeInMilliSecs'].values[0]
                        anomaly_stop = daily_symbol_anomalies[daily_symbol_anomalies['ClusterNo'] == cluster]['TimeInMilliSecs'].values[-1]

                        prev_stop = daily_symbol_anomalies[daily_symbol_anomalies['ClusterNo'] == clusters[i]]['TimeInMilliSecs'].values[-1]

                        # If subsequent frauds are separated by less than 70 seconds, than their neighborhoods are
                        # join to form a single sequence to use in the test set.
                        if anomaly_start - prev_stop > 2 * 35 * 1000:
                            anomalies_stops.append(prev_stop + 35 * 1000)
                            anomalies_starts.append(anomaly_start - 35 * 1000)

                        if cluster == clusters[-1]:
                            anomalies_stops.append(anomaly_stop + 35 * 1000)

                else:
                    anomaly_start = daily_symbol_anomalies['TimeInMilliSecs'].values[0] - 35 * 1000
                    anomaly_stop = daily_symbol_anomalies['TimeInMilliSecs'].values[-1] + 35 * 1000
                    anomalies_starts.append(anomaly_start)
                    anomalies_stops.append(anomaly_stop)

                # Getting all the sequential data from the previously defined anomaly starts and stops and sorting them
                # into the the normal and anomalous data sets.
                if len(anomalies_starts) == 1:
                    anomalous = daily_symbol_data.loc[(daily_symbol_data['TimeInMilliSecs'] >= anomalies_starts[0]) & (
                                daily_symbol_data['TimeInMilliSecs'] <= anomalies_stops[0])]
                    anomalous['Date'] = anomalous['Date'].astype(str) + '-' + str(sequence_idx)
                    anomalous_data.append(anomalous)
                    sequence_idx += 1

                    normal_before = daily_symbol_data.loc[daily_symbol_data['TimeInMilliSecs'] < anomalies_starts[0]]
                    normal_before['Date'] = normal_before['Date'].astype(str) + '-' + str(sequence_idx)
                    normal_data.append(normal_before)
                    sequence_idx += 1

                    normal_after = daily_symbol_data.loc[daily_symbol_data['TimeInMilliSecs'] > anomalies_stops[0]]
                    normal_after['Date'] = normal_after['Date'].astype(str) + '-' + str(sequence_idx)
                    normal_data.append(normal_after)
                    sequence_idx += 1

                else:
                    for anomaly, (anomaly_start, anomaly_stop) in enumerate(zip(anomalies_starts, anomalies_stops)):
                        anomalous = daily_symbol_data.loc[(daily_symbol_data['TimeInMilliSecs'] >= anomaly_start) & (
                                daily_symbol_data['TimeInMilliSecs'] <= anomaly_stop)]
                        anomalous['Date'] = anomalous['Date'].astype(str) + '-' + str(sequence_idx)
                        anomalous_data.append(anomalous)
                        sequence_idx += 1

                        if anomaly == 0:
                            normal_before = daily_symbol_data.loc[daily_symbol_data['TimeInMilliSecs'] < anomaly_start]
                            normal_before['Date'] = normal_before['Date'].astype(str) + '-' + str(sequence_idx)
                            normal_data.append(normal_before)
                            sequence_idx += 1

                        elif anomaly == len(anomalies_starts) - 1:
                            normal_before = daily_symbol_data.loc[(daily_symbol_data['TimeInMilliSecs'] < anomaly_start)
                                                                  & (daily_symbol_data['TimeInMilliSecs'] > anomalies_stops[anomaly - 1])]
                            normal_before['Date'] = normal_before['Date'].astype(str) + '-' + str(sequence_idx)
                            normal_data.append(normal_before)
                            sequence_idx += 1

                            normal_after = daily_symbol_data.loc[daily_symbol_data['TimeInMilliSecs'] > anomaly_stop]
                            normal_after['Date'] = normal_after['Date'].astype(str) + '-' + str(sequence_idx)
                            normal_data.append(normal_after)
                            sequence_idx += 1
                        else:
                            normal_between = daily_symbol_data.loc[(daily_symbol_data['TimeInMilliSecs'] < anomaly_start)
                                                                   & (daily_symbol_data['TimeInMilliSecs'] > anomalies_stops[anomaly - 1])]
                            normal_between['Date'] = normal_between['Date'].astype(str) + '-' + str(sequence_idx)
                            normal_data.append(normal_between)
                            sequence_idx += 1

            else:
                daily_symbol_data['Date'] = daily_symbol_data['Date'].astype(str) + '-' + str(sequence_idx)
                normal_data.append(daily_symbol_data)
                sequence_idx += 1

    # Concatenating all normal and anomalous subsequences into single data frames.
    normal_data = pd.concat(normal_data).sort_values(['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']).reset_index(drop=True)
    anomalous_data = pd.concat(anomalous_data).sort_values(['Date', 'Time', 'MilliSeconds', 'OriginalSequenceNumber']).reset_index(drop=True)

    # Shuffling normal data
    groups = [df for _, df in normal_data.groupby('Date')]
    random.shuffle(groups)
    normal_data = pd.concat(groups).reset_index(drop=True)

    # Splitting normal data into train and valid. The anomalous data set is used as the test set.
    train_data_stop = int(np.ceil(0.7 * normal_data.shape[0]))
    train_data_stop_date = normal_data.iloc[train_data_stop]['Date']
    train_data_stop_seq = normal_data[normal_data['Date'] == train_data_stop_date].index.values[-1]
    train_data = normal_data.iloc[:train_data_stop_seq]
    valid_data = normal_data.iloc[train_data_stop_seq:]

    return {'train': train_data, 'valid': valid_data, 'test': anomalous_data}
