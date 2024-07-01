import os
import math
import pickle

from tqdm import tqdm
from task1 import ASSETS_PATH, QUERY_FILE, CANDIDATE_FILE, Tokeniser, load_data
from task2 import load_inverted_index, load_passage_lengths


# File paths
TFIDF_OUTPUT = 'tfidf.csv'
BM25_OUTPUT = 'bm25.csv'
QUERY_SPARSE_TF = os.path.join(ASSETS_PATH, '_task3_query_sparse_tf.pkl')


def argsort(seq, reverse=False):
    """Argsort of a list"""
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def get_sparse_tfidf(sparse_tf, idf_map):
    """Convert sparse TF representation to sparce TF-IDF"""
    return {term: freq * idf_map[term] for term, freq in sparse_tf.items()}


def get_qid_order():
    qid_order = []
    print(f"Parsing qid order from {QUERY_FILE}...")
    with open(QUERY_FILE, 'r') as f:
        for line in tqdm(f.readlines()):
            qid, _ = line.split('\t')
            qid_order.append(qid)
    return qid_order


def get_qid2candidates():
    qid2candidates = {}
    print(f"Parsing qid2candidates from {CANDIDATE_FILE}...")
    with open(CANDIDATE_FILE, 'r') as f:
        for line in tqdm(f.readlines()):
            qid, pid, _, _ = line.split('\t')
            if qid not in qid2candidates:
                qid2candidates[qid] = [pid]
            else:
                qid2candidates[qid].append(pid)
    return qid2candidates


def get_query_sparse_tf(inverted_index):
    """Compute the sparse TF representation of queries"""
    query_sparse_tf = {}
    tokeniser = Tokeniser()
    print(f'Computing TF-IDF of queries from {QUERY_FILE}...')
    with open(QUERY_FILE, 'r') as f:
        for line in tqdm(f.readlines()):
            qid, query = line.split('\t')
            terms_freq = {}
            for token in tokeniser(query):
                if token not in inverted_index:  # skip OOV terms
                    continue
                if token not in terms_freq:
                    terms_freq[token] = 0
                terms_freq[token] += 1
            query_sparse_tf[qid] = terms_freq
    # Dump the query sparse TF for later tasks
    with open(QUERY_SPARSE_TF, 'wb+') as f:
        pickle.dump(query_sparse_tf, f)
    return query_sparse_tf


def load_query_sparse_tf():
    """For use in later tasks"""
    return load_data(binary=QUERY_SPARSE_TF,
                     fallback_func=lambda: get_query_sparse_tf(load_inverted_index()))


def cosine_similarity(vec, mat):
    """Compute cosine similarity between a sparse vector and
    each sparse vector in a matrix"""
    vec_norm = sum([v**2 for v in vec.values()])
    mat_norms = [sum([v**2 for v in m.values()]) for m in mat]
    dot_prods = [sum([vec[term] * m[term] for term in vec if term in m])
                 for m in mat]
    return [dot_prods[i] / math.sqrt(vec_norm * mat_norms[i])
            for i in range(len(mat))]


def bm25(query_tf, pid, inverted_index, passage_lengths, average_length,
         k1=1.2, k2=100, b=0.75):
    score = 0
    K = k1 * ((1-b) + b * passage_lengths[pid] / average_length)
    N = len(passage_lengths)
    for term, qf in query_tf.items():
        if pid not in inverted_index[term]:
            continue
        f = inverted_index[term][pid]
        n = len(inverted_index[term])
        score += (math.log10((N-n+0.5)/(n+0.5))
                  * ((k1+1)*f / (K+f))
                  * ((k2+1)*qf / (k2+qf)))
    return score


def task3():
    """Task 3 of the coursework 1."""
    print('-'*50 + '\nTask 3\n' + '-'*50)

    inverted_index = load_inverted_index()
    passage_lengths = load_passage_lengths()
    average_length = sum(passage_lengths.values()) / len(passage_lengths)

    ### TF of passages
    passage_sparse_tf = {}  # sparse representations
    idf_map = {}
    n_passage = len(passage_lengths)
    print('Computing TF-IDF of passages from inverted index..')
    for term, doc_freq in tqdm(inverted_index.items()):
        # IDF
        idf_map[term] = math.log10(n_passage / len(doc_freq))
        # TF
        for pid, freq in doc_freq.items():
            if pid not in passage_sparse_tf:
                passage_sparse_tf[pid] = {}
            passage_sparse_tf[pid][term] = freq

    ### TF of queries
    query_sparse_tf = get_query_sparse_tf(inverted_index)

    ### Parse queries and candidates info
    qid_order = get_qid_order()
    qid2candidates = get_qid2candidates()

    ### Cosine similarity
    print(f'Exporting TF-IDF query results to {TFIDF_OUTPUT}...')
    with open(TFIDF_OUTPUT, 'w') as f:
        start_line = ''  # empty for the first line
        for qid in tqdm(qid_order):
            pids = qid2candidates[qid]
            q_tfidf = get_sparse_tfidf(query_sparse_tf[qid], idf_map)
            p_tfidf = [get_sparse_tfidf(passage_sparse_tf[pid], idf_map)
                       for pid in pids]
            scores = cosine_similarity(q_tfidf, p_tfidf)
            top100 = argsort(scores, reverse=True)[:100]  # Descending order
            for arg in top100:
                f.write(start_line)
                f.write(f'{qid},{pids[arg]},{scores[arg]}')
                start_line = '\n'

    ### BM25
    print(f'Exporting BM25 query results to {BM25_OUTPUT}...')
    with open(BM25_OUTPUT, 'w') as f:
        start_line = ''  # empty for the first line
        for qid in tqdm(qid_order):
            pids = qid2candidates[qid]
            scores = [bm25(query_sparse_tf[qid], pid, inverted_index,
                           passage_lengths, average_length)
                      for pid in pids]
            top100 = argsort(scores, reverse=True)[:100]  # Descending order
            for arg in top100:
                f.write(start_line)
                f.write(f'{qid},{pids[arg]},{scores[arg]}')
                start_line = '\n'


if __name__ == '__main__':
    task3()
