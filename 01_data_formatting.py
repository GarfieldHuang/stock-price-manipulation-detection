"""
Formats LOBSTER data and adds synthesized trade-based manipulations.
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join


def add_quote_stuffing(data, anomalies, n, max_order_size):
    # Getting last anomaly cluster number
    try:
        previous_last_cluster = np.sort(anomalies['ClusterNo'].unique())[-1]
    except IndexError:
        previous_last_cluster = 0

    current_cluster = previous_last_cluster + 1

    anomalies_stuffing = pd.DataFrame(columns=anomalies.columns)

    # Finding time intervals longer where the bid-ask spread is greater or equal to 0.02 (or greater than 9bps)
    time_delta = data['TimeInMilliSecs'].diff()
    possible_starting_indexes = data[(data['AskPrice1'] / data['BidPrice1'] >= 1.0009) & (pd.isna(data['ClusterNo'])) & (pd.isna(data['ClusterNo'].shift(1))) & (pd.isna(data['ClusterNo'].shift(-1)))].index.values - 1

    # Bid quote-stuffing
    # Selecting random n//2 events
    bid_starting_indexes = np.sort(np.random.choice(possible_starting_indexes, size=n//2, replace=False))

    # Updating possible starting indexes
    possible_starting_indexes = np.array(list(set(possible_starting_indexes).difference(set(bid_starting_indexes))))

    # Bid columns
    bid_columns = []
    for i in range(1, 2):
        bid_columns.append('BidPrice' + str(i))
        bid_columns.append('BidSize' + str(i))

    # Creating bid quote-stuffing orders
    for index in bid_starting_indexes:
        starting_data = data.iloc[index]
        starting_bid_side = starting_data[bid_columns]

        # Creating random quote-stuffing length between 50 and 200 events at a random rate between 8 to 10 per ms.
        length = np.random.randint(low=50, high=200)
        rate = np.random.randint(low=8, high=10)
        spread = np.floor(0.5 * (starting_data['AskPrice1'] - starting_data['BidPrice1']) * 100)
        if spread > 1:
            delta = spread / 100
        else:
            delta = 0.01

        # Random order size
        size = np.random.randint(1, max_order_size+1)

        quote_stuffing = pd.DataFrame()
        for j in range(length):
            # Send new limit order at best bid + 1 tick
            if j % 2 == 0:
                new_bid_side = starting_bid_side.copy()
                new_bid_side[bid_columns[2:]] = starting_bid_side[bid_columns[:-2]].values
                new_bid_side['BidPrice1'] = starting_bid_side['BidPrice1'] + delta
                new_bid_side['BidSize1'] = size

                # Adding event into sequence
                new_data = starting_data.copy()
                new_data[bid_columns] = new_bid_side
                new_data['TradeIndicator'] = 0
                new_data['TradeSize'] = 0
                new_data['CancelledBidIndicator'] = 0
                new_data['CancelledAskIndicator'] = 0
                new_data['CancelledBidSize'] = 0
                new_data['CancelledAskSize'] = 0
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'].astype(str) + '-' + ("000000" + str(index))[-6:] + '-' + ("00" + str(j))[-3:]
                new_data['TimeInMilliSecs'] += j // rate
                minutes = ("0" + new_data['Minutes'].astype(str))[-2:]
                seconds = ("0" + new_data['Seconds'].astype(str))[-2:]
                new_data['Time'] = new_data['Hours'].astype(str) + minutes + seconds
                quote_stuffing = quote_stuffing.append(new_data)

            # Cancelling previous order
            else:
                # Adding event into sequence
                new_data = starting_data.copy()
                new_data['TradeIndicator'] = 0
                new_data['TradeSize'] = 0
                new_data['CancelledAskIndicator'] = 0
                new_data['CancelledAskSize'] = 0
                new_data['CancelledBidIndicator'] = 1
                new_data['CancelledBidSize'] = size
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'].astype(str) + '-' + ("000000" + str(index))[-6:] + '-' + ("00" + str(j))[-3:]
                new_data['TimeInMilliSecs'] += j // rate
                minutes = ("0" + new_data['Minutes'].astype(str))[-2:]
                seconds = ("0" + new_data['Seconds'].astype(str))[-2:]
                new_data['Time'] = new_data['Hours'].astype(str) + minutes + seconds
                quote_stuffing = quote_stuffing.append(new_data)

        quote_stuffing['ClusterNo'] = current_cluster
        current_cluster += 1
        anomalies_stuffing = anomalies_stuffing.append(quote_stuffing)

    # Ask quote-stuffing
    # Selecting random n//2 events
    ask_starting_indexes = np.sort(np.random.choice(possible_starting_indexes, size=n // 2, replace=False))

    # Ask columns
    ask_columns = []
    for i in range(1, 2):
        ask_columns.append('AskPrice' + str(i))
        ask_columns.append('AskSize' + str(i))

    # Creating ask quote-stuffing orders
    for index in ask_starting_indexes:
        starting_data = data.iloc[index]
        starting_ask_side = starting_data[ask_columns]

        # Creating random quote-stuffing length between 50 and 200 events at a random rate between 8 to 10 per ms.
        length = np.random.randint(low=50, high=200)
        rate = np.random.randint(low=8, high=10)
        spread = np.floor(0.5 * (starting_data['AskPrice1'] - starting_data['BidPrice1']) * 100)
        if spread > 1:
            delta = spread / 100
        else:
            delta = 0.01

        # Random order size
        size = np.random.randint(1, max_order_size + 1)

        quote_stuffing = pd.DataFrame()
        for j in range(length):
            # Send new limit order at best ask - 1 tick
            if j % 2 == 0:
                new_ask_side = starting_ask_side.copy()
                new_ask_side[ask_columns[2:]] = starting_ask_side[ask_columns[:-2]].values
                new_ask_side['AskPrice1'] = starting_ask_side['AskPrice1'] - delta
                new_ask_side['AskSize1'] = size

                # Adding event into sequence
                new_data = starting_data.copy()
                new_data[ask_columns] = new_ask_side
                new_data['TradeIndicator'] = 0
                new_data['TradeSize'] = 0
                new_data['CancelledBidIndicator'] = 0
                new_data['CancelledAskIndicator'] = 0
                new_data['CancelledBidSize'] = 0
                new_data['CancelledAskSize'] = 0
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'].astype(str) + '-' + ("000000" + str(index))[-6:] + '-' + ("00" + str(j))[-3:]
                new_data['TimeInMilliSecs'] += j // rate
                minutes = ("0" + new_data['Minutes'].astype(str))[-2:]
                seconds = ("0" + new_data['Seconds'].astype(str))[-2:]
                new_data['Time'] = new_data['Hours'].astype(str) + minutes + seconds
                quote_stuffing = quote_stuffing.append(new_data)

            # Cancelling previous order
            else:
                # Adding event into sequence
                new_data = starting_data.copy()
                new_data['TradeIndicator'] = 0
                new_data['TradeSize'] = 0
                new_data['CancelledBidIndicator'] = 0
                new_data['CancelledBidSize'] = 0
                new_data['CancelledAskIndicator'] = 1
                new_data['CancelledAskSize'] = size
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'].astype(str) + '-' + ("000000" + str(index))[-6:] + '-' + ("00" + str(j))[-3:]
                new_data['TimeInMilliSecs'] += j // rate
                minutes = ("0" + new_data['Minutes'].astype(str))[-2:]
                seconds = ("0" + new_data['Seconds'].astype(str))[-2:]
                new_data['Time'] = new_data['Hours'].astype(str) + minutes + seconds
                quote_stuffing = quote_stuffing.append(new_data)

        quote_stuffing['ClusterNo'] = current_cluster
        current_cluster += 1
        anomalies_stuffing = anomalies_stuffing.append(quote_stuffing)

    # Combining anomalies with real data
    anomalies_stuffing['FraudType'] = 1
    data['OriginalSequenceNumber'] = data['OriginalSequenceNumber'].astype(str)
    data = pd.concat((data, anomalies_stuffing), axis=0).sort_values(by=['index', 'OriginalSequenceNumber']).reset_index(drop=True)

    # Adjusting timestamps to account for insertion of synthetic data
    for cluster in range(int(current_cluster - n), int(current_cluster)):
        cluster_data = data[data['ClusterNo'] == cluster]
        cluster_data_indexes = cluster_data.index.values.astype(int)
        cluster_start_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[0]] - 1
        cluster_end_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[-1]]
        duration = cluster_end_time - cluster_start_time

        # Adding time delta to data occuring after cluster to account for synthetized data
        data.loc[cluster_data_indexes[-1] + 1:, 'TimeInMilliSecs'] += duration

    # Recomputing time values
    data['Hours'] = (data['TimeInMilliSecs'] // (1000 * 3600)).astype(int)
    data['Minutes'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600) // (1000 * 60)).astype(int)
    data['Seconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data['Minutes'] * 1000 * 60) // 1000).astype(int)
    data['MilliSeconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data['Minutes'] * 1000 * 60 - data['Seconds'] * 1000)).astype(int)
    minutes = ("0" + data['Minutes'].astype(str)).str[-2:]
    seconds = ("0" + data['Seconds'].astype(str)).str[-2:]
    data['Time'] = data['Hours'].astype(str) + minutes + seconds

    # Getting anomalies with modified timestamps
    anomalies = data[~pd.isna(data['ClusterNo'])]

    return data, anomalies


def add_layering(data, anomalies, n, mean_bid_order, mean_ask_order):
    # Getting last anomaly cluster number
    try:
        previous_last_cluster = np.sort(anomalies['ClusterNo'].unique())[-1]
    except IndexError:
        previous_last_cluster = 0

    current_cluster = previous_last_cluster + 1
    ticker = data['ExternalSymbol'].unique()[0]

    anomalies_layering = pd.DataFrame(columns=anomalies.columns)

    # Finding where the bid-ask spread is larger than 8bps
    possible_starting_indexes = data[(data['AskPrice1'] / data['BidPrice1'] >= 1.0008) & (pd.isna(data['ClusterNo'])) & (pd.isna(data['ClusterNo'].shift(1))) & (pd.isna(data['ClusterNo'].shift(-1)))].index.values - 1

    if possible_starting_indexes.shape[0] == 0:
        return data, anomalies

    # Bid Layering
    # Selecting random n//2 events for bid layering
    bid_starting_indexes = np.sort(np.random.choice(possible_starting_indexes, size=n // 2, replace=False))

    # Updating possible starting indexes
    possible_starting_indexes = np.array(list(set(possible_starting_indexes).difference(set(bid_starting_indexes))))

    ask_starting_indexes = np.sort(np.random.choice(possible_starting_indexes, size=n // 2, replace=False))

    # Bid columns
    bid_columns = []
    for i in range(1, 2):
        bid_columns.append('BidPrice' + str(i))
        bid_columns.append('BidSize' + str(i))

    # Ask columns
    ask_columns = []
    for i in range(1, 2):
        ask_columns.append('AskPrice' + str(i))
        ask_columns.append('AskSize' + str(i))

    # Creating layering orders
    for index in bid_starting_indexes:
        starting_data = data.iloc[index]
        starting_bid_side = starting_data[bid_columns]
        starting_ask_side = starting_data[ask_columns]
        max_bid_price = np.round(np.minimum(np.ceil(starting_data['BidPrice1'] * 1.0007 * 100) / 100, starting_data['AskPrice1'] - 0.01), 2)
        min_ask_price = np.round(np.maximum(np.floor(starting_data['AskPrice1'] * 0.9997 * 100) / 100, max_bid_price + 0.01), 2)
        bona_fide_size = np.ceil(mean_ask_order * np.random.uniform(2, 3))
        # Sending sell order
        layering = pd.DataFrame()
        new_data = starting_data.copy()
        new_data[ask_columns[2:]] = starting_ask_side[ask_columns[:-2]].values
        new_data['AskPrice1'] = min_ask_price
        new_data['AskSize1'] = bona_fide_size + starting_data['AskSize1'] * (min_ask_price == starting_data['AskPrice1'])
        new_data['TradeIndicator'] = 0
        new_data['TradeSize'] = 0
        new_data['CancelledBidIndicator'] = 0
        new_data['CancelledAskIndicator'] = 0
        new_data['CancelledBidSize'] = 0
        new_data['CancelledAskSize'] = 0
        new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'] + '-' + ("000000" + str(index))[-6:] + '-' + "000"
        new_data['TimeInMilliSecs'] += 1
        layering = layering.append(new_data)

        # Selecting length, price increment and rate of orders per level of price upgrade that best achieves increase in bid price
        fit = np.inf
        best_length = 0
        best_price_increment = 0
        best_rate = 0
        for length in [10, 11, 12]:
            price_increment = np.maximum(np.floor(((max_bid_price - starting_data['BidPrice1']) / length) * 100) / 100, 0.01)
            rate = np.ceil(length / np.maximum((max_bid_price - starting_data['BidPrice1']) / price_increment, 1))
            length_fit = (max_bid_price - starting_data['BidPrice1']) - np.ceil((length / rate)) * price_increment
            if length_fit < fit:
                best_length = length
                best_price_increment = price_increment
                best_rate = rate
                fit = length_fit

        # Sending non-bona fide bid orders
        order_size = 100 * np.ceil((mean_bid_order * 6 // length) / 100)
        order_size += np.ceil(order_size * np.random.uniform(-0.1, 0.1))
        for j in range(best_length):
            # Send new limit order at best bid + delta
            new_data = new_data.copy()
            new_data[bid_columns[2:]] = new_data[bid_columns[:-2]].values
            new_data['BidPrice1'] = new_data['BidPrice1'] + best_price_increment * ((j % best_rate) == 0)
            new_data['BidSize1'] = order_size + order_size * (j % best_rate)

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(j + 1))[-3:]
            new_data['TimeInMilliSecs'] += 14 + np.random.choice([_ for _ in range(-5, 6)])
            layering = layering.append(new_data)

        # Adding a trade of bona fide sell order 10ms after last non-bona fide order
        new_data = new_data.copy()
        new_data[ask_columns] = starting_ask_side[ask_columns].values
        new_data['TradeIndicator'] = -1
        new_data['TradeSize'] = -bona_fide_size
        new_data['CancelledBidIndicator'] = 0
        new_data['CancelledAskIndicator'] = 0
        new_data['CancelledBidSize'] = 0
        new_data['CancelledAskSize'] = 0
        new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(length + 1))[-3:]
        new_data['TimeInMilliSecs'] += 10 + np.random.choice([_ for _ in range(-5, 6)])
        layering = layering.append(new_data)

        # Cancelling non-bona-fide orders after 600ms
        for j in range(best_length):
            new_data = new_data.copy()
            new_data['BidPrice1'] = layering['BidPrice1'].values[-3 - 2 * j]
            new_data['BidSize1'] = layering['BidSize1'].values[-3 - 2 * j]

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 1
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = order_size
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(length + j + 2))[-3:]
            if j == 0:
                new_data['TimeInMilliSecs'] += 600 + 100 * np.random.choice([_ for _ in range(-5, 5)])
            layering = layering.append(new_data)

        layering['ClusterNo'] = current_cluster
        current_cluster += 1
        anomalies_layering = anomalies_layering.append(layering)

    # Ask layering
    # Bid columns
    bid_columns = []
    for i in range(1, 2):
        bid_columns.append('BidPrice' + str(i))
        bid_columns.append('BidSize' + str(i))

    # Ask columns
    ask_columns = []
    for i in range(1, 2):
        ask_columns.append('AskPrice' + str(i))
        ask_columns.append('AskSize' + str(i))

    # Creating layering orders
    for index in ask_starting_indexes:
        starting_data = data.iloc[index]
        starting_bid_side = starting_data[bid_columns]
        starting_ask_side = starting_data[ask_columns]
        min_ask_price = np.round(np.maximum(np.floor(starting_data['AskPrice1'] * 0.9993 * 100) / 100, starting_data['BidPrice1'] + 0.01), 2)
        max_bid_price = np.round(np.minimum(np.ceil(starting_data['BidPrice1'] * 1.0003 * 100) / 100, min_ask_price - 0.01), 2)
        bona_fide_size = np.ceil(mean_ask_order * np.random.uniform(2, 3))

        # Sending buy order
        layering = pd.DataFrame()
        new_data = starting_data.copy()
        new_data[bid_columns[2:]] = starting_bid_side[bid_columns[:-2]].values
        new_data['BidPrice1'] = max_bid_price
        new_data['BidSize1'] = bona_fide_size + starting_data['BidSize1'] * (max_bid_price == starting_data['BidPrice1'])
        new_data['TradeIndicator'] = 0
        new_data['TradeSize'] = 0
        new_data['CancelledBidIndicator'] = 0
        new_data['CancelledAskIndicator'] = 0
        new_data['CancelledBidSize'] = 0
        new_data['CancelledAskSize'] = 0
        new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'] + '-' + ("000000" + str(index))[-6:] + '-' + "000"
        new_data['TimeInMilliSecs'] += 1
        layering = layering.append(new_data)

        # Selecting length, price increment and rate of orders per level of price upgrade that best achieves decrease in ask price
        fit = np.inf
        best_length = 0
        best_price_increment = 0
        best_rate = 0
        for length in [10, 11, 12]:
            price_increment = np.maximum(np.floor(((starting_data['AskPrice1'] - min_ask_price) / length) * 100) / 100, 0.01)
            rate = np.ceil(length / np.maximum((starting_data['AskPrice1'] - min_ask_price) / price_increment, 1))
            length_fit = (starting_data['AskPrice1'] - min_ask_price) - np.ceil((length / rate)) * price_increment
            if length_fit < fit:
                best_length = length
                best_price_increment = price_increment
                best_rate = rate
                fit = length_fit

        # Sending non-bona fide ask orders
        order_size = 100 * np.ceil((mean_ask_order * 6 // length) / 100)
        order_size += np.ceil(order_size * np.random.uniform(-0.1, 0.1))
        for j in range(best_length):
            # Send new limit order at best ask - delta
            new_data = new_data.copy()
            new_data[ask_columns[2:]] = new_data[ask_columns[:-2]].values
            new_data['AskPrice1'] = new_data['AskPrice1'] - best_price_increment * ((j % best_rate) == 0)
            new_data['AskSize1'] = order_size + order_size * (j % best_rate)

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(j + 1))[-3:]
            new_data['TimeInMilliSecs'] += 14 + np.random.choice([_ for _ in range(-5, 6)])
            layering = layering.append(new_data)

        # Adding a trade of bona fide buy order 10ms after last non-bona fide order
        new_data = new_data.copy()
        new_data[bid_columns] = starting_bid_side[bid_columns].values
        new_data['TradeIndicator'] = 1
        new_data['TradeSize'] = bona_fide_size
        new_data['CancelledBidIndicator'] = 0
        new_data['CancelledAskIndicator'] = 0
        new_data['CancelledBidSize'] = 0
        new_data['CancelledAskSize'] = 0
        new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(length + 1))[-3:]
        new_data['TimeInMilliSecs'] += 10 + np.random.choice([_ for _ in range(-5, 6)])
        layering = layering.append(new_data)

        # Cancelling non-bona-fide orders after 600ms
        for j in range(best_length):
            new_data = new_data.copy()
            new_data['AskPrice1'] = layering['AskPrice1'].values[-3 - 2 * j]
            new_data['AskSize1'] = layering['AskSize1'].values[-3 - 2 * j]

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 1
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = order_size
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(length + j + 2))[-3:]
            if j == 0:
                new_data['TimeInMilliSecs'] += 600 + 100 * np.random.choice([_ for _ in range(-5, 5)])
            layering = layering.append(new_data)

        layering['ClusterNo'] = current_cluster
        current_cluster += 1
        anomalies_layering = anomalies_layering.append(layering)

    # Combining anomalies with real data
    anomalies_layering['FraudType'] = 2
    data['OriginalSequenceNumber'] = data['OriginalSequenceNumber'].astype(str)
    data = pd.concat((data, anomalies_layering), axis=0).sort_values(by=['index', 'OriginalSequenceNumber']).reset_index(drop=True)

    # Adjusting timestamps to account for insertion of synthetic data
    for cluster in range(int(current_cluster - n), int(current_cluster)):
        cluster_data = data[data['ClusterNo'] == cluster]
        cluster_data_indexes = cluster_data.index.values.astype(int)
        cluster_start_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[0]] - 1
        cluster_end_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[-1]]
        duration = cluster_end_time - cluster_start_time

        # Adding time delta to data occuring after cluster to account for synthetized data
        data.loc[cluster_data_indexes[-1] + 1:, 'TimeInMilliSecs'] += duration

    # Recomputing time values
    data['Hours'] = (data['TimeInMilliSecs'] // (1000 * 3600)).astype(int)
    data['Minutes'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600) // (1000 * 60)).astype(int)
    data['Seconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data['Minutes'] * 1000 * 60) // 1000).astype(int)
    data['MilliSeconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data['Minutes'] * 1000 * 60 -data['Seconds'] * 1000)).astype(int)
    minutes = ("0" + data['Minutes'].astype(str)).str[-2:]
    seconds = ("0" + data['Seconds'].astype(str)).str[-2:]
    data['Time'] = data['Hours'].astype(str) + minutes + seconds

    # Getting anomalies with modified timestamps
    anomalies = data[~pd.isna(data['ClusterNo'])]

    return data, anomalies


def add_pnd(data, anomalies, n, mean_volume):
    # Getting last anomaly cluster number
    try:
        previous_last_cluster = np.sort(anomalies['ClusterNo'].unique())[-1]
    except IndexError:
        previous_last_cluster = 0

    current_cluster = previous_last_cluster + 1

    anomalies_pnd = pd.DataFrame(columns=anomalies.columns)

    # Finding n random possible starting indexes
    possible_starting_indexes = data[(pd.isna(data['ClusterNo'])) & (pd.isna(data['ClusterNo'].shift(1))) &
                                     (pd.isna(data['ClusterNo'].shift(-1)))].index.values - 1
    starting_indexes = np.sort(np.random.choice(possible_starting_indexes, size=n, replace=False))

    # Bid columns
    bid_columns = []
    for i in range(1, 2):
        bid_columns.append('BidPrice' + str(i))
        bid_columns.append('BidSize' + str(i))

    # Ask columns
    ask_columns = []
    for i in range(1, 2):
        ask_columns.append('AskPrice' + str(i))
        ask_columns.append('AskSize' + str(i))

    # Creating pump-and-dump sequences
    for index in starting_indexes:
        new_data = data.iloc[index]

        price_delta = np.random.uniform(0.03, 0.04)
        price_delta_per_order = np.ceil(100 * new_data['AskPrice1'] * price_delta / 6) / 100  # 6 bid and ask orders to push prices
        volume_per_order = np.ceil(np.random.uniform(1.25, 2) * mean_volume)
        pumping = pd.DataFrame()

        # Sending pumping orders
        for j in range(6):
            # Send new ask limit order at best ask + delta
            new_data = new_data.copy()
            new_data['AskPrice1'] = new_data['AskPrice1'] + price_delta_per_order
            new_data['AskSize1'] = volume_per_order

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = 0
            if j == 0:
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'] + '-' + ("000000" + str(index))[-6:] + '-000'
            else:
                new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(2 * j))[-3:]
            new_data['TimeInMilliSecs'] += np.floor(1000 / 12) + np.floor(np.random.uniform(-20, 20))  # random time interval between each pumping order
            pumping = pumping.append(new_data)

            # Send new buy limit order at best bid + delta
            new_data = new_data.copy()
            new_data['BidPrice1'] = new_data['BidPrice1'] + price_delta_per_order
            new_data['BidSize1'] = volume_per_order

            # Adding event into sequence
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(2 * j + 1))[-3:]
            new_data['TimeInMilliSecs'] += np.floor(1000 / 12) + np.floor(np.random.uniform(-20, 20)) # random time intervail between each pumping order
            pumping = pumping.append(new_data)

        # Cancelling bid orders and adding trades on the ask
        for j in range(6):
            new_data = new_data.copy()
            new_data['BidPrice1'] = new_data['BidPrice1'] - price_delta_per_order
            new_data['BidSize1'] = volume_per_order

            # Adding bid cancellation
            new_data['TradeIndicator'] = 0
            new_data['TradeSize'] = 0
            new_data['CancelledBidIndicator'] = 1
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = volume_per_order
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(2 * j + 12))[-3:]
            new_data['TimeInMilliSecs'] += np.floor(3000 / 12) + np.floor(np.random.uniform(-60, 60))
            pumping = pumping.append(new_data)

            new_data = new_data.copy()
            new_data['AskPrice1'] = new_data['AskPrice1'] - price_delta_per_order
            new_data['AskSize1'] = volume_per_order

            # Adding ask trade
            new_data['TradeIndicator'] = -1
            new_data['TradeSize'] = -volume_per_order
            new_data['CancelledBidIndicator'] = 0
            new_data['CancelledAskIndicator'] = 0
            new_data['CancelledBidSize'] = 0
            new_data['CancelledAskSize'] = 0
            new_data['OriginalSequenceNumber'] = new_data['OriginalSequenceNumber'][:-3] + ("00" + str(2 * j + 1 + 12))[-3:]
            new_data['TimeInMilliSecs'] += np.floor(3000 / 12) + np.floor(np.random.uniform(-60, 60))
            pumping = pumping.append(new_data)

        pumping['ClusterNo'] = current_cluster
        current_cluster += 1
        anomalies_pnd = anomalies_pnd.append(pumping)

    # Combining anomalies with real data
    anomalies_pnd['FraudType'] = 3
    data['OriginalSequenceNumber'] = data['OriginalSequenceNumber'].astype(str)
    data = pd.concat((data, anomalies_pnd), axis=0).sort_values(by=['index', 'OriginalSequenceNumber']).reset_index(drop=True)

    # Adjusting timestamps to account for insertion of synthetic data
    for cluster in range(int(current_cluster - n), int(current_cluster)):
        cluster_data = data[data['ClusterNo'] == cluster]
        cluster_data_indexes = cluster_data.index.values.astype(int)
        cluster_start_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[0]] - 1
        cluster_end_time = cluster_data['TimeInMilliSecs'][cluster_data_indexes[-1]]
        duration = cluster_end_time - cluster_start_time

        # Adding time delta to data occuring after cluster to account for synthetized data
        data.loc[cluster_data_indexes[-1] + 1:, 'TimeInMilliSecs'] += duration

    # Recomputing time values
    data['Hours'] = (data['TimeInMilliSecs'] // (1000 * 3600)).astype(int)
    data['Minutes'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600) // (1000 * 60)).astype(int)
    data['Seconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data[
        'Minutes'] * 1000 * 60) // 1000).astype(int)
    data['MilliSeconds'] = ((data['TimeInMilliSecs'] - data['Hours'] * 1000 * 3600 - data['Minutes'] * 1000 * 60 - data['Seconds'] * 1000)).astype(int)
    minutes = ("0" + data['Minutes'].astype(str)).str[-2:]
    seconds = ("0" + data['Seconds'].astype(str)).str[-2:]
    data['Time'] = data['Hours'].astype(str) + minutes + seconds

    # Getting anomalies with modified timestamps
    anomalies = data[~pd.isna(data['ClusterNo'])]

    return data, anomalies


if __name__ == "__main__":
    # Getting all stock tickers
    path = "data\\"
    files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".csv")]
    stocks = set([file[:4] for file in files])

    for stock in stocks:
        print(stock)
        # Importing data sets
        messages = pd.read_csv(path + stock + "_2012-06-21_34200000_57600000_message_1.csv", header=None)
        lobs = pd.read_csv(path + stock + "_2012-06-21_34200000_57600000_orderbook_1.csv", header=None)

        messages.columns = ['time', 'type', 'id', 'size', 'price', 'direction']
        lobs_columns = []
        lobs_columns.append('AskPrice1')
        lobs_columns.append('AskSize1')
        lobs_columns.append('BidPrice1')
        lobs_columns.append('BidSize1')
        lobs.columns = lobs_columns

        price_columns = []
        price_columns.append('AskPrice1')
        price_columns.append('BidPrice1')

        size_columns = []
        size_columns.append('AskSize1')
        size_columns.append('BidSize1')

        lobs[price_columns] /= 10000

        # Combining LOB data with messages
        stock_data = pd.concat((messages, lobs), axis=1)
        stock_data = stock_data[stock_data['type'] != 5].reset_index() # get rid of hidden executions
        stock_data['Date'] = 20120621
        stock_data['ExternalSymbol'] = stock
        stock_data['GroupStatus'] = 'Continuous trading'
        stock_data['OriginalSequenceNumber'] = stock_data['id']
        stock_data['time'] *= 1000000

        # Reworking time
        stock_data['Hours'] = (stock_data['time'] // (1000000 * 3600)).astype(int)
        stock_data['Minutes'] = ((stock_data['time'] - stock_data['Hours'] * 1000000 * 3600) // (1000000 * 60)).astype(int)
        stock_data['Seconds'] = ((stock_data['time'] - stock_data['Hours'] * 1000000 * 3600 - stock_data['Minutes'] * 1000000 * 60) // 1000000).astype(int)
        stock_data['MilliSeconds'] = ((stock_data['time'] - stock_data['Hours'] * 1000000 * 3600 - stock_data['Minutes'] * 1000000 * 60 - stock_data['Seconds'] * 1000000) // 1000 + 1).astype(int)

        minutes = ("0" + stock_data['Minutes'].astype(str)).str[-2:]
        seconds = ("0" + stock_data['Seconds'].astype(str)).str[-2:]
        stock_data['Time'] = (stock_data['Hours'].astype(str) + minutes + seconds).astype(int)
        stock_data['TimeInMilliSecs'] = stock_data['Hours'] * 3600 * 1000 + stock_data['Minutes'] * 60 * 1000 + stock_data['Seconds'] * 1000 + stock_data['MilliSeconds']

        # Creating a trade indicator and trade size columns
        stock_data['TradeIndicator'] = 0
        stock_data.loc[stock_data['type'] == 4, 'TradeIndicator'] = stock_data.loc[(stock_data['type'] == 4), 'direction']
        stock_data['TradeSize'] = (stock_data['TradeIndicator']) * stock_data['size']

        # Creating order cancellations indicators
        stock_data['CancelledBidIndicator'] = 0
        stock_data['CancelledAskIndicator'] = 0
        stock_data['CancelledBidSize'] = 0
        stock_data['CancelledAskSize'] = 0

        stock_data.loc[((stock_data['type'] == 2) | (stock_data['type'] == 3)) & (stock_data['direction'] == 1), 'CancelledBidIndicator'] = 1
        stock_data.loc[((stock_data['type'] == 2) | (stock_data['type'] == 3)) & (stock_data['direction'] == -1), 'CancelledAskIndicator'] = 1
        stock_data['CancelledBidSize'] = (stock_data['CancelledBidIndicator']) * stock_data['size']
        stock_data['CancelledAskSize'] = (stock_data['CancelledAskIndicator']) * stock_data['size']

        # Computing the mean order size on both sides of LOB, and mean L1 size
        mean_bid_order = stock_data.loc[(stock_data['type'] == 1) & (stock_data['direction'] == 1), 'size'].mean()
        mean_ask_order = stock_data.loc[(stock_data['type'] == 1) & (stock_data['direction'] == -1), 'size'].mean()
        max_order_size = np.floor(np.percentile(stock_data.loc[stock_data['type'] == 1, 'size'].values, q=10))
        mean_volume = 0.5 * (stock_data['BidSize1'].mean() + stock_data['AskSize1'].mean())

        # Dropping unnecessary columns and adding columns for fraud insertion
        stock_data = stock_data.drop(columns=['time', 'type', 'id', 'size', 'price', 'direction'])
        stock_data['ClusterNo'] = np.nan
        stock_data['FraudType'] = np.nan

        # Anomaly dataframe
        anomalies = pd.DataFrame(columns=stock_data.columns)

        # Quote stuffing
        # Adding 50 quote stuffing frauds (25 on best bid and 25 on best ask)
        stock_data, anomalies = add_quote_stuffing(stock_data, anomalies, 50, max_order_size)

        # Layering
        # Adding 50 layering frauds (25 on best bid and 25 on best ask)
        stock_data, anomalies = add_layering(stock_data, anomalies, 50, mean_bid_order, mean_ask_order)

        # Pump-and-dump
        # Adding 50 pump-and-dump frauds
        stock_data, anomalies = add_pnd(stock_data, anomalies, 50, mean_volume=mean_volume)

        # Getting all anomalies in a single dataframe
        anomalies = stock_data.loc[~pd.isna(stock_data['FraudType']), ['Date', 'ExternalSymbol', 'Time', 'MilliSeconds',
                                                                       'TimeInMilliSecs', 'OriginalSequenceNumber',
                                                                       'ClusterNo', 'FraudType']]

        # Saving the formatted stock data and anomalies
        stock_data.to_csv(path + "formatted\\" + stock + ".csv", index=False, sep=';')
        anomalies.to_csv(path + "formatted\\" + stock + "_Anomalies.csv", index=False, sep=';')

    # Combining data into single file
    path = r"data\formatted\\"
    files_market = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith(".csv") and "Anomalies" not in f)]
    files_market.sort()
    files_anomalies = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith(".csv") and "Anomalies" in f)]
    files_anomalies.sort()
    all_data = pd.DataFrame()
    anomalies = pd.DataFrame()

    # Correcting ClusterNo to have unique values for each anomaly
    stock_nb = 0
    for file in files_market:
        stock_df = pd.read_csv(path + file, sep=";")
        stock_df['ClusterNo'] += stock_nb * 150
        all_data = all_data.append(stock_df, ignore_index=True)
        stock_nb += 1

    stock_nb = 0
    for file in files_anomalies:
        stock_df = pd.read_csv(path + file, sep=";")
        stock_df['ClusterNo'] += stock_nb * 150
        anomalies = anomalies.append(stock_df, ignore_index=True)
        stock_nb += 1

    # Reformatting anomalies and combining all data
    anomalies = anomalies.sort_values(by=['TimeInMilliSecs', 'OriginalSequenceNumber'])
    anomalies['OriginalSequenceNumber'] = anomalies['OriginalSequenceNumber'].astype(str)
    anomalies = anomalies[['Date', 'ExternalSymbol', 'Time', 'MilliSeconds', 'OriginalSequenceNumber', 'ClusterNo', 'FraudType']]
    anomalies.to_csv(path + "Anomalies_formatted.csv", index=False, sep=';')

    all_data = all_data.sort_values(by=['index', 'TimeInMilliSecs', 'OriginalSequenceNumber'])
    all_data['OriginalSequenceNumber'] = all_data['OriginalSequenceNumber'].astype(str)
    all_data.to_csv(path + "MarketDepth_20120621.csv", index=False, sep=';')