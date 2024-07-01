import itertools

import numpy as np
import xgboost as xgb
from tqdm import tqdm

from parameters import XGB_PARAMS_GRID, XGB_N_FOLD, XGB_NUM_BOOST_ROUND
from utils import (Standardizer, get_processed_data, evaluate_validation_data,
                   evaluate_test_data)


def lambda_mart_features(q_embs, p_embs, standardizer, train=False, label=None, qid=None):
    """Use cosine similarity and dot product as feature. Standardize
    the feature as model input."""
    dot_prod = np.sum(q_embs * p_embs, axis=1, keepdims=True)
    q_norm = np.sqrt(np.sum(q_embs**2, axis=1, keepdims=True))
    p_norm = np.sqrt(np.sum(p_embs**2, axis=1, keepdims=True))
    feat = np.hstack([dot_prod, q_norm, p_norm])
    if train:
        stand_feat = standardizer.fit_transform(feat)
    else:
        stand_feat = standardizer.transform(feat)
    return xgb.DMatrix(stand_feat, label=label, qid=qid)


def xgb_ndcg_ranker_cv(params_iter, d_matrix, n_fold, num_boost_round):
    """Grid Search Cross validation for XGBoost Ranker."""
    best_score = 0
    best_params = None

    # Loop over parameters
    for params in tqdm(params_iter):
        params['objective'] = 'rank:ndcg'  # Ranking objective
        params['eval_metric'] = 'ndcg'     # CV metrics

        #  Cross validation
        cv_results = xgb.cv(params, d_matrix, num_boost_round=num_boost_round,
                            nfold=n_fold, as_pandas=True)

        # Update best model
        mean_ndcg = cv_results['test-ndcg-mean'].mean()
        if mean_ndcg > best_score:
            best_score = mean_ndcg
            best_params = params

    return best_params


def params_iterator(params_grid):
    keys = list(params_grid.keys())
    values = [params_grid[k] for k in keys]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def task3():
    print('-'*50 + '\nTask 3\n' + '-'*50)

    # Load embeddings and labels
    print('## Loading training data embeddings ...')
    tr_ids, tr_q_embs, tr_p_embs, tr_y = get_processed_data('train')  # Training data
    print('## Loading validation data embeddings ...')
    va_ids, va_q_embs, va_p_embs, va_y = get_processed_data('valid')  # Validation data

    # Order training data by qid (required by xgboost)
    tr_qids = tr_ids[:, 0].astype(int)
    tr_idx = np.argsort(tr_qids)
    tr_qids = tr_qids[tr_idx]
    tr_q_embs = tr_q_embs[tr_idx]
    tr_p_embs = tr_p_embs[tr_idx]
    tr_y = tr_y[tr_idx]

    # Process model input features
    print('## Processing model inputs features ...')
    standardizer = Standardizer()
    tr_x = lambda_mart_features(tr_q_embs, tr_p_embs, standardizer,
                                train=True, label=tr_y, qid=tr_qids)
    va_x = lambda_mart_features(va_q_embs, va_p_embs, standardizer)

    # Cross validation
    params_iter = params_iterator(XGB_PARAMS_GRID)
    print(f'## Hyperparameter searching over {len(params_iter)} models ...')
    best_params = xgb_ndcg_ranker_cv(params_iter, tr_x, n_fold=XGB_N_FOLD,
                                     num_boost_round=XGB_NUM_BOOST_ROUND)

    # Best model
    print('## Training the best model with full training data ...')
    best_model = xgb.train(best_params, tr_x)

    # Evaluate validation data
    print('## Evaluating best models on validation set ...')
    evaluate_validation_data(va_ids[:, 0], va_x, va_y, best_model.predict,
                             'LM', best_params)

    # Evaluate test data
    print('## Evaluating best models on testing set ...')
    get_test_feat = lambda x1, x2: lambda_mart_features(x1, x2, standardizer)
    evaluate_test_data(best_model.predict, model_name='LM', feat_func=get_test_feat)

    print('## Task 3 completed!')


if __name__ == '__main__':
    task3()