import os
import json
import pickle
import torch
import torchtext
import numpy as np
import pandas as pd
from tqdm import tqdm

from parameters import IRREL_KEEP, GLOVE_MODEL, GLOVE_EMBEDDING_DIM


DATA_DIR = 'data'
ASSET_DIR = 'assets'
os.makedirs(ASSET_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.tsv')
VALID_FILE = os.path.join(DATA_DIR, 'validation_data.tsv')
QUERY_FILE = os.path.join(DATA_DIR, 'test-queries.tsv')
PASSAGE_FILE = os.path.join(DATA_DIR, 'passage_collection_new.txt')
CANDIDATE_FILE = os.path.join(DATA_DIR, 'candidate_passages_top1000.tsv')

DATA_FILE = {'train': TRAIN_FILE, 'valid': VALID_FILE, 'test': CANDIDATE_FILE}


# ---------------------------------
# Data Preprocessing
# ---------------------------------

def get_data_df(type):
    """Load the required data as a pandas DataFrame."""
    if type in ['train', 'valid']:
        df = pd.read_csv(DATA_FILE[type], sep='\t', skiprows=1,
                         names=['qid', 'pid', 'query', 'passage', 'relevancy'],
                         dtype={'pid': str, 'qid': str, 'relevancy': int})
    elif type == 'test':
        df = pd.read_csv(DATA_FILE[type], sep='\t',
                         dtype={'qid': str, 'pid': str},
                         names=['qid', 'pid', 'query', 'passage'])
    return df.sort_values('qid')


def get_processed_data(type, force_compute=False):
    """Load or compute embeddings for the required set of data."""

    # Check if embeddings are already computed
    if type == 'train':
        file = os.path.join(ASSET_DIR, f'processed_train_{IRREL_KEEP}.pkl')
    else:
        file = os.path.join(ASSET_DIR, f'processed_{type}.pkl')
    if os.path.exists(file) and not force_compute:
        with open(file, 'rb') as f:
            return pickle.load(f)

    # Load data
    if type == 'train':
        # down-sample irrelevant records to balance classes
        df = down_sample_irrel(get_data_df(type))
    else:
        df = get_data_df(type)

    # Compute embeddings
    tokeniser = get_tokeniser()
    emb_model = get_embedding_model()
    ids = df[['qid', 'pid']].values
    q_embs = doc_embedding(df['query'], tokeniser, emb_model)
    p_embs = doc_embedding(df['passage'], tokeniser, emb_model)

    # Labels
    if type == 'test':
        labels = np.zeros(len(ids))  # dummy labels
    else:
        labels = df['relevancy'].values

    # Save as pickle
    with open(file, 'wb') as f:
        pickle.dump((ids, q_embs, p_embs, labels), f)

    return ids, q_embs, p_embs, labels


def down_sample_irrel(df):
    """Keep only some irrelevant passages for each query."""

    def cut_irrel(df):
        rel_df = df[df['relevancy'] == 1]  # keep all relevant passages
        irrel_df = df[df['relevancy'] == 0]
        reduced = irrel_df.sample(min(len(irrel_df), IRREL_KEEP), random_state=11)
        return pd.concat([rel_df, reduced])

    return df.groupby('qid').apply(cut_irrel, include_groups=False).reset_index()


def doc_embedding(texts, tokeniser, emb_model):
    """Tokenise and compute document embeddings by averaging all word
    embeddings in the document."""
    text_embs = []
    # for text in tqdm(texts):
    for text in tqdm(texts):
        # Average the token embeddings as the document embedding
        token_embs = emb_model.get_vecs_by_tokens(tokeniser(text))
        text_embs.append(torch.mean(token_embs, dim=0))
    return torch.vstack(text_embs).numpy()


def get_embedding_model():
    """Load and download GloVe embeddings."""
    cache = os.path.join(ASSET_DIR, '.vector_cache')
    return torchtext.vocab.GloVe(GLOVE_MODEL, GLOVE_EMBEDDING_DIM, cache=cache)


def get_tokeniser():
    return torchtext.data.utils.get_tokenizer('basic_english')


class Standardizer:
    """Transform feature with zero mean and unit variance."""

    def __init__(self):
        self.means = None
        self.stdvs = None

    def fit(self, data):
        self.means = np.mean(data, axis=0)
        self.stdvs = np.std(data, axis=0)

    def transform(self, data):
        if self.means is None or self.stdvs is None:
            raise ValueError('The standardizer has not been fit yet!')
        return (data - self.means) / self.stdvs

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


# ---------------------------------
# Evaluation
# ---------------------------------

def ndcg(y_pred, k=None):
    """Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        y_pred (list): a list of true relevancy in predicted rank order
        k (int): rank cutoff
    """

    def dcg(rels, k=None):
        if k is None:
            k = len(rels)
        k = min(k, len(rels))
        rels = np.array(rels)[:k]
        return np.sum(rels / np.log2(np.arange(2, k + 2)))

    y_pred = np.array(y_pred)
    y_true = np.sort(y_pred)[::-1]

    return dcg(y_pred, k) / dcg(y_true, k)  # DCG / IDCG


def avg_precision(y_pred, k=None):
    """Average Precision (AP) at rank k.

    Args:
        y_pred (list): a list of true relevancy in predicted rank order
        k (int): rank cutoff
    """
    if k is None:
        k = len(y_pred)
    k = min(k, len(y_pred))

    y_pred = np.array(y_pred)[:k]
    total_rel = np.sum(y_pred)

    if total_rel == 0:
        return 0

    precision_k = np.cumsum(y_pred) / np.arange(1, k + 1)
    relevancy_k = y_pred
    return np.sum(precision_k * relevancy_k) / total_rel


def f1_score(y_true, y_pred):
    """F1 score."""
    true_pos = np.sum(np.array(y_true) * np.array(y_pred))
    precision = true_pos / np.sum(y_pred)
    recall = true_pos / np.sum(y_true)
    return 2 * precision * recall / (precision + recall)


def evaluate_validation_data(va_qids, va_x, va_y, pred_func, model_name, specs=None):
    """Evaluate the validation data and export the result for report use."""

    def ap_agg(df, k=None):
        return avg_precision(df['true'].values, k)

    def ndcg_agg(df, k=None):
        return ndcg(df['true'].values, k)

    if model_name not in ['BM25', 'LR', 'LM', 'NN']:
        raise ValueError('Model name not supported')

    # Predict
    va_df = pd.DataFrame({'qid': va_qids, 'true': va_y, 'pred': pred_func(va_x)})
    # Sort by predicted score
    va_df = va_df.sort_values('pred', ascending=False)
    qid_group = va_df.groupby('qid')
    # Metrics
    specs = specs or {}
    for cf in [3, 10, 100, 1000]:  # Rank cut-offs
        specs[f'mAP@{cf}'] = qid_group.apply(lambda x: ap_agg(x, k=cf),
                                             include_groups=False).mean()
        specs[f'mNDCG@{cf}'] = qid_group.apply(lambda x: ndcg_agg(x, k=cf),
                                               include_groups=False).mean()
    # Export the result
    file = os.path.join(ASSET_DIR, f'{model_name}_metrics.json')
    specs['irrel_keep'] = IRREL_KEEP
    with open(file, 'w') as f:
        json.dump(specs, f, indent=4)
    print(f'Validation metrics saved to {file}')
    return specs


def evaluate_test_data(model, model_name, feat_func):
    """Evaluate the test data and export the result for submission."""

    def top100(df):
        df = df.sort_values('score', ascending=False).head(100)
        df['rank'] = range(1, len(df) + 1)
        return df

    if model_name not in ['LR', 'LM', 'NN']:
        raise ValueError('Invalid model name!')
    # Load test data
    te_ids, te_q_embs, te_p_embs, _ = get_processed_data('test')
    te_X = feat_func(te_q_embs, te_p_embs)
    # Predict
    te_df = pd.DataFrame({'qid': te_ids[:, 0], 'pid': te_ids[:, 1]})
    te_df['score'] = model(te_X)
    # Order query as in 'test-queries.tsv'
    qid_order = pd.read_csv(QUERY_FILE, sep='\t', names=['qid'], dtype=str, usecols=[0])
    order_df = qid_order.merge(te_df, on='qid', how='left')
    # Keep only the top 100 passages for each query
    score_df = order_df.groupby('qid', sort=False) \
        .apply(top100, include_groups=False)
    # Format and export the result
    score_df['qid'] = order_df['qid']
    score_df['a2'] = 'A2'
    score_df['model'] = model_name
    score_df.to_csv(f'{model_name}.txt', sep=' ', index=True, header=False,
                    columns=['qid', 'a2', 'pid', 'rank', 'score', 'model'])
    print(f'Test data result saved to {model_name}.txt')
