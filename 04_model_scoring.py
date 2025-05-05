import pandas as pd
from scoring.utilities import *
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics import auc
np.random.seed(54321)
torch.random.manual_seed(54321)


def compute_confusion_stats(test_data, ids):
    """
    Computes number of True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN) based
    on flagged ids on test_data.

    Parameters
    ----------
    test_data : pandas df
        Test data set on which to compute metrics
    ids: list of int
        List of ids detected by a model

    Returns
    -------
    tp: int
        Number of true positives
    tn: int
        Number of true negatives
    fp: int
        Number of false positives
    fn: int
        Number of false negatives
    """
    # Adding flagged ids to test_data
    ids = np.unique(ids)
    test_data['Flag'] = 0
    test_data.loc[test_data['index'].isin(ids), 'Flag'] = 1

    # Computing tp, tn, fp and fn
    tp = test_data[(test_data['Fraud'] == 1) & (test_data['Flag'] == 1)].shape[0]
    tn = test_data[(test_data['Fraud'] == 0) & (test_data['Flag'] == 0)].shape[0]
    fp = test_data[(test_data['Fraud'] == 0) & (test_data['Flag'] == 1)].shape[0]
    fn = test_data[(test_data['Fraud'] == 1) & (test_data['Flag'] == 0)].shape[0]

    return tp, tn, fp, fn


def dissimilarity_score(model, train_dataset, valid_dataset, test_dataset, features, s, latent_dim=128):
    """
    Computes all out-of-sample performance metrics of a given dissimilarity-based model.

    Parameters
    ----------
    model : str
        Path to a PyTorch model
    train_dataset : str
        Path to the train set
    valid_dataset : str
        Path to the valid set
    test_dataset : str
        Path to the test set
    features :  list of str
        List of features
    s : int
        Sequence length of model
    latent_dim : int
        Latent dimension of model

    Returns
    -------
    None
    """
    # Computing latent vectors of all data sets
    input_features = features.copy()
    input_features.append('FraudType')
    print("Computing latent vectors on train set...")
    train_set, _, _ = generating_latent_vector(model, train_dataset, input_features, s, latent_dim)

    print("Computing latent vectors on valid set...")
    valid_set, _, _ = generating_latent_vector(model, valid_dataset, input_features, s, latent_dim)
    train_valid_set = np.concatenate((train_set, valid_set))
    del train_set, valid_set, _

    input_features2 = features.copy()
    input_features2.append('index')
    input_features2.append('FraudType')
    print("Computing latent vectors on test set...")
    test_set, fraudulent_test, ids = generating_latent_vector(model, test_dataset, input_features2, s, latent_dim,
                                                              index=True)
    print("Computing performance on test set...")

    # Grouping frauds and defining their ids
    clean_set = test_set[fraudulent_test == 0]
    fraud_set_type1 = test_set[fraudulent_test == 1]
    fraud_set_type2 = test_set[fraudulent_test == 2]
    fraud_set_type3 = test_set[fraudulent_test == 3]
    del test_set

    ids_test = ids[fraudulent_test == 0, :]
    ids_fraud_set_type1 = ids[fraudulent_test == 1, :]
    ids_fraud_set_type2 = ids[fraudulent_test == 2, :]
    ids_fraud_set_type3 = ids[fraudulent_test == 3, :]
    del fraudulent_test, ids

    all_ids = np.concatenate((ids_test, ids_fraud_set_type1, ids_fraud_set_type2, ids_fraud_set_type3))

    # Training the OC-SVM on the fraud-free set consisting in the concatenation of the train and valid sets
    nb_dim = train_valid_set.shape[0]
    var = train_valid_set.var()
    gamma = 10 / (nb_dim * var)
    transform = Nystroem(gamma=gamma, random_state=42)
    clf_sgd = SGDOneClassSVM(shuffle=True, fit_intercept=True, random_state=42, tol=1e-4)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    oc_classifier = pipe_sgd.fit(train_valid_set)
    del train_valid_set

    # Dissimilarity scoring of the sequences in test set
    scores_test_clean = -oc_classifier.score_samples(clean_set)
    scores_fraud_type1 = -oc_classifier.score_samples(fraud_set_type1)
    scores_fraud_type2 = -oc_classifier.score_samples(fraud_set_type2)
    scores_fraud_type3 = -oc_classifier.score_samples(fraud_set_type3)

    # Merging test set with dissimilarity scores
    test_set = pd.read_parquet(test_dataset, engine='fastparquet')[['ExternalSymbol', 'Date', 'TimeInMilliSecs',
                                                                    'OriginalSequenceNumber', 'Fraud', 'ClusterNo',
                                                                    'index']]

    scores = np.concatenate((scores_test_clean, scores_fraud_type1, scores_fraud_type2, scores_fraud_type3))
    scores = scores.reshape((-1, 1))
    scores = np.repeat(scores, repeats=all_ids.shape[1], axis=1)
    scores_df = pd.DataFrame({'index': all_ids.flatten(), 'score': scores.flatten()}).groupby('index').mean().\
        reset_index()
    test_set = test_set.merge(scores_df, how='left', on='index')

    # Computing the performance metrics for every decision threshold tau on a general basis
    precisions = []
    recalls = []
    tprs = []
    fprs = []
    max_threshold = test_set['score'].max()
    min_threshold = test_set['score'].min()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    for threshold in thresholds:
        ids = test_set.loc[test_set['score'] > threshold, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        tpr = recall
        fpr = fp / (fp + tn)

        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)

    roc = auc(fprs, tprs)
    print("AUROC: " + str(round(roc, 3)))

    pr = auc(recalls, precisions)
    print("AUC-PR: " + str(round(pr, 3)))

    beta = 4
    f_scores = np.array([(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall
                         in zip(precisions, recalls) if precision + recall != 0])
    f_score = np.max(f_scores)
    precision = precisions[np.argmax(f_scores)]
    recall = recalls[np.argmax(f_scores)]
    print('F score: ' + str(round(f_score, 3)))
    print('Precision: ' + str(round(precision, 3)))
    print('Recall: ' + str(round(recall, 3)))
    print()

    # Finding best tau
    best_tau = thresholds[np.argmax(f_scores)]

    # Computing the metrics on a per-stock basis with the best tau
    for stock in test_set['ExternalSymbol'].unique():
        test_set_stock = test_set[test_set['ExternalSymbol'] == stock]
        precisions = []
        recalls = []

        ids = test_set.loc[test_set['score'] > best_tau, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set_stock.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        precisions.append(precision)
        recalls.append(recall)

        beta = 4
        f_scores = np.array(
            [(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall in
             zip(precisions, recalls) if precision + recall != 0])
        f_score = np.max(f_scores)
        precision = precisions[np.argmax(f_scores)]
        recall = recalls[np.argmax(f_scores)]
        print(stock + ' F score: ' + str(round(f_score, 3)))
        print(stock + ' Precision: ' + str(round(precision, 3)))
        print(stock + ' Recall: ' + str(round(recall, 3)))
        print()

    return


def reconstruction_score(model, test_dataset, features, s):
    """
    Computes all out-of-sample performance metrics of a given reconstruction-based model.

    Parameters
    ----------
    model : str
        Path to a PyTorch model
    test_dataset : str
        Path to the test set
    features :  list of str
        List of features
    s : int
        Sequence length of model

    Returns
    -------
    None
    """
    # Computing errors of test set subsequences
    input_features = features.copy()
    input_features.append('index')
    input_features.append('FraudType')
    print("Computing error vectors on test set...")
    error_vectors_test, fraudulent_test, ids = generating_error_vector(model, test_dataset, input_features, s)

    print("Computing performance on test set...")

    # Grouping frauds and defining their ids
    scores_test_clean = error_vectors_test[fraudulent_test == 0]
    scores_fraud_type1 = error_vectors_test[fraudulent_test == 1]
    scores_fraud_type2 = error_vectors_test[fraudulent_test == 2]
    scores_fraud_type3 = error_vectors_test[fraudulent_test == 3]
    del error_vectors_test

    ids_test = ids[fraudulent_test == 0, :]
    ids_fraud_set_type1 = ids[fraudulent_test == 1, :]
    ids_fraud_set_type2 = ids[fraudulent_test == 2, :]
    ids_fraud_set_type3 = ids[fraudulent_test == 3, :]
    all_ids = np.concatenate((ids_test, ids_fraud_set_type1, ids_fraud_set_type2, ids_fraud_set_type3))
    del ids

    # Merging test set with error scores
    test_set = pd.read_parquet(test_dataset, engine='fastparquet')[['ExternalSymbol', 'Date', 'TimeInMilliSecs',
                                                                    'OriginalSequenceNumber', 'Fraud', 'ClusterNo',
                                                                    'index']]

    scores = np.concatenate((scores_test_clean, scores_fraud_type1, scores_fraud_type2, scores_fraud_type3))
    scores = scores.reshape((-1, 1))
    scores = np.repeat(scores, repeats=all_ids.shape[1], axis=1)
    scores_df = pd.DataFrame({'index': all_ids.flatten(), 'score': scores.flatten()}).groupby('index').mean(). \
        reset_index()
    test_set = test_set.merge(scores_df, how='left', on='index')

    # Computing the performance metrics for every decision threshold tau on a general basis
    precisions = []
    recalls = []
    tprs = []
    fprs = []
    max_threshold = test_set['score'].max()
    min_threshold = test_set['score'].min()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    for threshold in thresholds:
        ids = test_set.loc[test_set['score'] > threshold, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        tpr = recall
        fpr = fp / (fp + tn)

        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)

    roc = auc(fprs, tprs)
    print("AUROC: " + str(round(roc, 3)))

    pr = auc(recalls, precisions)
    print("AUC-PR: " + str(round(pr, 3)))

    beta = 4
    f_scores = np.array([(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall
                         in zip(precisions, recalls) if precision + recall != 0])
    f_score = np.max(f_scores)
    precision = precisions[np.argmax(f_scores)]
    recall = recalls[np.argmax(f_scores)]
    print('F score: ' + str(round(f_score, 3)))
    print('Precision: ' + str(round(precision, 3)))
    print('Recall: ' + str(round(recall, 3)))
    print()

    # Finding best tau
    best_tau = thresholds[np.argmax(f_scores)]

    # Computing the metrics on a per-stock basis with the best tau
    for stock in test_set['ExternalSymbol'].unique():
        test_set_stock = test_set[test_set['ExternalSymbol'] == stock]
        precisions = []
        recalls = []

        ids = test_set.loc[test_set['score'] > best_tau, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set_stock.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        precisions.append(precision)
        recalls.append(recall)

        beta = 4
        f_scores = np.array(
            [(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall in
             zip(precisions, recalls) if precision + recall != 0])
        f_score = np.max(f_scores)
        precision = precisions[np.argmax(f_scores)]
        recall = recalls[np.argmax(f_scores)]
        print(stock + ' F score: ' + str(round(f_score, 3)))
        print(stock + ' Precision: ' + str(round(precision, 3)))
        print(stock + ' Recall: ' + str(round(recall, 3)))
        print()

    return


def oc_svm(train_dataset, valid_dataset, test_dataset, features, s):
    """
    Computes all out-of-sample performance metrics for OC-SVM.

    Parameters
    ----------
    train_dataset : str
        Path to the train set
    valid_dataset : str
        Path to the valid set
    test_dataset : str
        Path to the test set
    features :  list of str
        List of features
    s : int
        Sequence length of model

    Returns
    -------
    None
    """
    # Computing latent vectors of all data sets
    input_features = features.copy()
    input_features.append('FraudType')

    print("Getting sequences on train set...")
    sequences_train, _, _ = generating_sequence_vector(train_dataset, input_features, s)

    print("Getting sequences on valid set...")
    sequences_valid, _, _ = generating_sequence_vector(valid_dataset, input_features, s)
    train_valid_set = np.concatenate((sequences_train, sequences_valid))
    del sequences_train, sequences_valid, _

    input_features2 = features.copy()
    input_features2.append('index')
    input_features2.append('FraudType')
    print("Getting sequences on test set...")
    sequences_test, fraudulent_test, ids = generating_sequence_vector(test_dataset, input_features2, s, index=True)

    print("Computing performance on test set...")

    test_set = sequences_test[fraudulent_test == 0]
    fraud_set_type1 = sequences_test[fraudulent_test == 1]
    fraud_set_type2 = sequences_test[fraudulent_test == 2]
    fraud_set_type3 = sequences_test[fraudulent_test == 3]
    del sequences_test

    ids_test = ids[fraudulent_test == 0, :]
    ids_fraud_set_type1 = ids[fraudulent_test == 1, :]
    ids_fraud_set_type2 = ids[fraudulent_test == 2, :]
    ids_fraud_set_type3 = ids[fraudulent_test == 3, :]
    all_ids = np.concatenate((ids_test, ids_fraud_set_type1, ids_fraud_set_type2, ids_fraud_set_type3))
    del fraudulent_test, ids

    nb_dim = train_valid_set.shape[0]
    var = train_valid_set.var()
    gamma = 10 / (nb_dim * var)
    transform = Nystroem(gamma=gamma, random_state=42)
    clf_sgd = SGDOneClassSVM(shuffle=True, fit_intercept=True, random_state=42, tol=1e-4)
    pipe_sgd = make_pipeline(transform, clf_sgd)
    oc_classifier = pipe_sgd.fit(train_valid_set)

    scores_test_clean = -oc_classifier.score_samples(test_set)
    scores_fraud_type1 = -oc_classifier.score_samples(fraud_set_type1)
    scores_fraud_type2 = -oc_classifier.score_samples(fraud_set_type2)
    scores_fraud_type3 = -oc_classifier.score_samples(fraud_set_type3)

    # Merging test set with error scores
    test_set = pd.read_parquet(test_dataset, engine='fastparquet')[['ExternalSymbol', 'Date', 'TimeInMilliSecs',
                                                                    'OriginalSequenceNumber', 'Fraud', 'ClusterNo',
                                                                    'index']]

    scores = np.concatenate((scores_test_clean, scores_fraud_type1, scores_fraud_type2, scores_fraud_type3))
    scores = scores.reshape((-1, 1))
    scores = np.repeat(scores, repeats=all_ids.shape[1], axis=1)
    scores_df = pd.DataFrame({'index': all_ids.flatten(), 'score': scores.flatten()}).groupby('index').mean(). \
        reset_index()
    test_set = test_set.merge(scores_df, how='left', on='index')

    # Computing the performance metrics for every decision threshold tau on a general basis
    precisions = []
    recalls = []
    tprs = []
    fprs = []
    max_threshold = test_set['score'].max()
    min_threshold = test_set['score'].min()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    for threshold in thresholds:
        ids = test_set.loc[test_set['score'] > threshold, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        tpr = recall
        fpr = fp / (fp + tn)

        precisions.append(precision)
        recalls.append(recall)
        tprs.append(tpr)
        fprs.append(fpr)

    roc = auc(fprs, tprs)
    print("AUROC: " + str(round(roc, 3)))

    pr = auc(recalls, precisions)
    print("AUC-PR: " + str(round(pr, 3)))

    beta = 4
    f_scores = np.array([(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall
                         in zip(precisions, recalls) if precision + recall != 0])
    f_score = np.max(f_scores)
    precision = precisions[np.argmax(f_scores)]
    recall = recalls[np.argmax(f_scores)]
    print('F score: ' + str(round(f_score, 3)))
    print('Precision: ' + str(round(precision, 3)))
    print('Recall: ' + str(round(recall, 3)))
    print()

    # Finding best tau
    best_tau = thresholds[np.argmax(f_scores)]

    # Computing the metrics on a per-stock basis with the best tau
    for stock in test_set['ExternalSymbol'].unique():
        test_set_stock = test_set[test_set['ExternalSymbol'] == stock]
        precisions = []
        recalls = []

        ids = test_set.loc[test_set['score'] > best_tau, 'index'].values
        tp, tn, fp, fn = compute_confusion_stats(test_set_stock.copy(), ids)

        recall = tp / (tp + fn)
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        precisions.append(precision)
        recalls.append(recall)

        beta = 4
        f_scores = np.array(
            [(1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall) for precision, recall in
             zip(precisions, recalls) if precision + recall != 0])
        f_score = np.max(f_scores)
        precision = precisions[np.argmax(f_scores)]
        recall = recalls[np.argmax(f_scores)]
        print(stock + ' F score: ' + str(round(f_score, 3)))
        print(stock + ' Precision: ' + str(round(precision, 3)))
        print(stock + ' Recall: ' + str(round(recall, 3)))
        print()

    return


if __name__ == '__main__':
    # User defined parameters
    train_data_path = r'data/preprocessed_lobster/train_data.parquet'
    valid_data_path = r'data/preprocessed_lobster/valid_data.parquet'
    test_data_path = r'data/preprocessed_lobster/test_data.parquet'
    frauds_data_path = r'data/formatted_lobster/Anomalies_formatted.csv'
    features = ['ReturnBid1', 'ReturnAsk1', 'DerivativeReturnBid1', 'DerivativeReturnAsk1', 'BidSize1', 'AskSize1',
                'TradeBidSize', 'TradeAskSize', 'CancelledBidSize', 'CancelledAskSize', 'TradeBidIndicator',
                'TradeAskIndicator', 'CancelledBidIndicator', 'CancelledAskIndicator']
    seq_length = 25
    model_path = r'models/best_TransformerAutoencoder.pt'
    # model_path = r'models/best_StackedLSTMAutoencoder.pt'
    # model_path = r'models/best_MLPAutoencoder.pt'

    # Model scoring
    dissimilarity_score(model_path, train_data_path, valid_data_path, test_data_path,
                        features, seq_length)

    # reconstruction_score(model_path, test_data_path, features, seq_length)

    # oc_svm(train_data_path, valid_data_path, test_data_path, features, seq_length)




