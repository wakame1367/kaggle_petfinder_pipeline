import os
from functools import partial
from math import sqrt
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from petfinder_pipeline.utils import is_script_running

# https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/utils.py#L17
ON_KAGGLE: bool = is_script_running()
DATA_ROOT = Path('../input/petfinder-adoption-prediction'
                 if ON_KAGGLE else '../resources/petfinder-adoption-prediction')


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator) / denominator


def load_datasets():
    train = pd.read_csv(DATA_ROOT / "train/train.csv")
    test = pd.read_csv(DATA_ROOT / "test/test.csv")
    breeds = pd.read_csv(DATA_ROOT / "breed_labels.csv")
    colors = pd.read_csv(DATA_ROOT / "color_labels.csv")
    states = pd.read_csv(DATA_ROOT / "state_labels.csv")
    return train, test, breeds, colors, states


def run_lgb(train, test, target, lgb_params, train_params):
    n_splits = 4
    kf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    all_coefficients = []
    for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(train, target)):
        print('Fold {}/{}'.format(fold_idx + 1, n_splits))
        trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

        model = lgb.train(
            lgb_params,
            trn_data,
            valid_names=['train', 'valid'],
            valid_sets=[trn_data, val_data],
            **train_params)

        oof[val_idx] = model.predict(train.iloc[val_idx], num_iteration=model.best_iteration)
        predictions += model.predict(test, num_iteration=model.best_iteration) / n_splits
        print("RMSE: {}".format(rmse(target.iloc[val_idx], oof[val_idx])))
        opt_r = OptimizedRounder()
        opt_r.fit(oof[val_idx], target.iloc[val_idx])
        coefficients = opt_r.coefficients()
        pred_test_y_k = opt_r.predict(oof[val_idx], coefficients)
        qwk = quadratic_weighted_kappa(target.iloc[val_idx], pred_test_y_k)
        print("QWK = ", qwk)
        all_coefficients.append(coefficients)
    return predictions, all_coefficients


def main():
    train, test, breeds, colors, states = load_datasets()
    target_col = 'AdoptionSpeed'
    target = train[target_col]
    train_id = train['PetID']
    test_id = test['PetID']
    train.drop([target_col, 'PetID'], axis=1, inplace=True)
    test.drop(['PetID'], axis=1, inplace=True)

    train_desc = train.Description.fillna("none").values
    test_desc = test.Description.fillna("none").values

    # make TFIDF features
    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Fit TFIDF
    tfv.fit(list(train_desc))
    train_tfv_features = tfv.transform(train_desc)
    test_tfv_features = tfv.transform(test_desc)

    # decrease features div
    svd = TruncatedSVD(n_components=120)
    svd.fit(train_tfv_features)
    train_tfv_features = svd.transform(train_tfv_features)
    test_tfv_features = svd.transform(test_tfv_features)

    svd_feature_cols = ['svd_{}'.format(i) for i in range(120)]
    train_tfv_features = pd.DataFrame(train_tfv_features, columns=svd_feature_cols)
    test_tfv_features = pd.DataFrame(test_tfv_features, columns=svd_feature_cols)
    train = pd.concat((train, train_tfv_features), axis=1)
    test = pd.concat((test, test_tfv_features), axis=1)

    drop_cols = ['Name', 'RescuerID', 'Description']

    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    numeric_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
                    'AdoptionSpeed'] + svd_feature_cols
    cat_cols = list(set(train.columns) - set(numeric_cols))
    train.loc[:, cat_cols] = train[cat_cols].astype('category')
    test.loc[:, cat_cols] = test[cat_cols].astype('category')

    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 70,
              'max_depth': 8,
              'learning_rate': 0.01,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8,
              'min_split_gain': 0.02,
              'min_child_samples': 150,
              'min_child_weight': 0.02,
              'lambda_l2': 0.0475,
              'verbosity': -1,
              'data_random_seed': 17}
    train_params = {'num_boost_round': 20000,
                    'early_stopping_rounds': 1000,
                    'verbose_eval': 1000}
    predicts, coefficients = run_lgb(train, test, target, params, train_params)
    opt_r = OptimizedRounder()
    coefficients_ = np.mean(coefficients, axis=0)
    # manually adjust coefs
    coefficients_[0] = 1.645
    coefficients_[1] = 2.115
    coefficients_[3] = 2.84
    predictions = opt_r.predict(predicts, coefficients_).astype(int)
    submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': predictions})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
