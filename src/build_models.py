"""Get word vectors."""

import re

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC


def get_model_metrics(y_true, y_pred):
    """Return a dictionary of relevant model metrics.

    Parameters
    ----------
    y_true: numpy.array
      array of actual y values
    y_pred: numpy.array
      array of predicted y values

    Returns
    -------
    dict
      dictionary including F1 score, precision, recall
    """
    return {'f1': metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            'recall': metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            'precision': metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted'),
            'accuracy': metrics.accuracy_score(y_true=y_true, y_pred=y_pred)}


def evaluate_classifier_with_cv(X, y, n_folds, clf, shuffled=False):
    """Run cross-validation with parameters."""
    model_performance = []
    kfolds = KFold(n_splits=n_folds, shuffle=True)
    for fold, (train_index, validation_index) in enumerate(kfolds.split(X)):
        X_train, X_valid = X[train_index], X[validation_index]
        y_train, y_valid = y[train_index], y[validation_index]
        clf.fit(X=X_train, y=y_train)
        # Make model predictions
        y_pred_train = clf.predict(X=X_train)
        y_pred_valid = clf.predict(X=X_valid)
        # Evaluate model
        training_metrics = get_model_metrics(y_train, y_pred_train)
        validation_metrics = get_model_metrics(y_valid, y_pred_valid)
        model_performance.append({
            'recall_train': training_metrics['recall'],
            'recall_valid': validation_metrics['recall'],
            'precision_train': training_metrics['precision'],
            'precision_valid': validation_metrics['precision'],
            'f1_train': training_metrics['f1'],
            'f1_valid': validation_metrics['f1'],
            'accuracy_train': training_metrics['accuracy'],
            'accuracy_valid': validation_metrics['accuracy'],
            'fold': fold,
            'shuffled': shuffled
        })
    df_model_performance = pd.DataFrame.from_dict(model_performance)
    return df_model_performance


def create_dataset(roots_to_syllables, model, syllable_component='nucleus'):
    """Return mappings from vectors to specified syllable component (e.g. 'nucleus')."""
    X, y, words = [], [], []
    for root, syllable in roots_to_syllables.items():
        if root in model:
            syl = syllable[syllable_component]
            if syl != '':
                X.append(model[root])
                y.append(syl)
                words.append(root)
    return np.array(X), np.array(y), words


def permutation_test(X, y, clf, n=3):
    """Get estimate of model accuracy on randomly shuffled data."""
    performances = []
    for shuffle in range(n):
        print(shuffle)
        y_shuffle = np.random.permutation(y)
        df_performance = evaluate_classifier_with_cv(
            X, y_shuffle, n_folds=2, clf=clf, shuffled=True)
        performances.append(df_performance)
    shuffle_performance = pd.concat(performances)
    return shuffle_performance


def get_model(model_name):
    mappings = {'linear_svc': LinearSVC}
    return mappings[model_name]()


def reduce_dimensionality(X, n=20):
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X)
    return X_reduced
