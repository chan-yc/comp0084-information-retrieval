import re
from collections import defaultdict

import nltk
import numpy as np
from tqdm import tqdm

from utils import VALID_FILE, evaluate_validation_data


nltk.download('stopwords')


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


class BM25Tokeniser:

    def __init__(self):
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self._preprocesses = [
            self._replace_non_alphanumeric,
            self._split_alphanumeric,
            self.stemmer.stem,
        ]

    def __call__(self, doc, remove_stopwords=True):
        for process in self._preprocesses:
            doc = process(doc)
        tokens = self._split_by_space(doc)
        if remove_stopwords:
            tokens = [t for t in tokens if not self.is_stop(t)]
        return tokens

    def is_stop(self, token):
        """Stopword defined by NLTK"""
        return token in self.stopwords

    @staticmethod
    def _replace_non_alphanumeric(doc):
        """Replace non-alphanumeric with space"""
        puntuations = re.compile(r'[^a-zA-Z0-9\s]')
        return re.sub(puntuations, ' ', doc)

    @staticmethod
    def _split_alphanumeric(doc):
        """Split concatenated alphanumeric"""
        doc = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', doc)  # Split 0a to 0 a
        doc = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', doc)  # Split a0 to a 0
        return doc

    @staticmethod
    def _split_by_space(doc):
        """Split by space"""
        return re.split(r'\s+', doc)


def bm25(query_tf, pid, inver_idx, pid2len, avg_doc_len, k1=1.2, k2=100,
         b=0.75):
    score = 0
    N = len(pid2len)  # corpus size
    K = k1 * ((1-b) + b * pid2len[pid] / avg_doc_len)
    for term, qf in query_tf.items():
        if pid not in inver_idx[term]:  # term not in passage
            continue
        pf = inver_idx[term][pid]
        n = len(inver_idx[term])  # number of docs with this term
        score += (np.log10((N-n+0.5) / (n+0.5))
                  * ((k1+1) * pf / (K+pf))
                  * ((k2+1) * qf / (k2+qf)))
    return score


def parse_validation_data():

    rel_map = {}    # {qid: {pid: relevancy}}
    qid2tf = {}     # {qid: {term: term freq}}
    pid2len = {}    # {pid: passage_length}
    inver_idx = {}  # {term: {pid: term freq}}
    qid2pids = {}   # {qid: [pid]}
    tokeniser = BM25Tokeniser()

    with open(VALID_FILE, 'r') as f:
        f.readline()  # Skip header
        file_iterator = tqdm(f, total=int(file_line_count(VALID_FILE))-1)

        for line in file_iterator:
            qid, pid, query, passage, relevancy = line.strip().split('\t')

            # Query candidates
            qid2pids[qid] = qid2pids.get(qid, []) + [pid]

            # Construct inverted index from all passages
            if pid not in pid2len:
                terms = tokeniser(passage)
                pid2len[pid] = len(terms)
                # Count passage term frequency
                for term in terms:
                    inver_idx.setdefault(term, defaultdict(int))
                    inver_idx[term][pid] += 1

            # Construct query term frequency
            if qid not in qid2tf:
                terms = tokeniser(query)
                # Count query term frequency
                qid2tf[qid] = {}
                for term in terms:
                    qid2tf[qid][term] = qid2tf[qid].get(term, 0) + 1

            # Construct relevancy map
            if qid not in rel_map:
                rel_map[qid] = {pid: float(relevancy)}
            elif pid not in rel_map[qid]:
                rel_map[qid][pid] = float(relevancy)

    # Remove OOV terms from queries
    for qid in qid2tf:
        qid2tf[qid] = {t: tf for t, tf in qid2tf[qid].items() if t in inver_idx}

    return rel_map, qid2tf, pid2len, inver_idx, qid2pids


def file_line_count(filepath):
    """Return the number of lines in a file."""
    import subprocess
    command = ['wc', '-l', filepath]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])


def task1():
    print('-'*50 + '\nTask 1\n' + '-'*50)

    # Parse validation data for BM25
    print('## Parsing validation data...')
    rel_map, qid2tf, pid2len, inver_idx, qid2pids = parse_validation_data()
    avg_doc_len = np.mean(list(pid2len.values()))

    # Run BM25 on validation data
    print('## Scoring queries\' passage candidates using BM25...')
    qids, true_rels, pred_rels = [], [], []
    # Loop over queries and corresponding passages
    for qid, query_tf in tqdm(qid2tf.items()):
        pid_candidates = qid2pids[qid]
        for pid in pid_candidates:
            bm25_prob = bm25(query_tf, pid, inver_idx, pid2len, avg_doc_len)
            qids.append(qid)
            true_rels.append(rel_map[qid][pid])
            pred_rels.append(bm25_prob)

    # Evaluate validation data
    print('## Evaluating the model on validation set ...')
    evaluate_validation_data(qids, None, true_rels, lambda _: pred_rels, 'BM25')

    print('## Task 1 completed!')


if __name__ == '__main__':
    task1()
