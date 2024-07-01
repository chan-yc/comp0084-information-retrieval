import math

from tqdm import tqdm
from task2 import load_inverted_index, load_passage_lengths
from task3 import load_query_sparse_tf, get_qid_order, get_qid2candidates, argsort


# File paths
LAPLACE_OUTPUT = 'laplace.csv'
LIDSTONE_OUTPUT = 'lidstone.csv'
DIRICHLET_OUTPUT = 'dirichlet.csv'


def lidstone_correction(query_tf, pid, inverted_index, N, **kwargs):
    eps = kwargs.get('eps', 1.0)
    score = 0
    for term, tf in query_tf.items():
        prob = ((inverted_index[term].get(pid, 0) + eps)
                / (N + eps * len(inverted_index)))
        score += tf * math.log(prob)
    return score


def dirichlet_smoothing(query_tf, pid, inverted_index, N, **kwargs):
    mu, total, term_freq_sum = kwargs['mu'], kwargs['total'], kwargs['term_freq_sum']
    score = 0
    for term, tf in query_tf.items():
        prob = N * inverted_index[term].get(pid, 0) / N  # Doc prob
        prob += mu * term_freq_sum[term] / total  # Collection prob
        score += tf * math.log(prob / (N + mu))
    return score


def task4():
    """Task 4 of the coursework 1."""
    print('-'*50 + '\nTask 4\n' + '-'*50)

    inverted_index = load_inverted_index()
    passage_lengths = load_passage_lengths()
    query_sparse_tf = load_query_sparse_tf()
    qid_order = get_qid_order()
    qid2candidates = get_qid2candidates()

    total = sum(passage_lengths.values())
    # mu = total / len(passage_lengths)  # average length
    term_freq_sum = {k: sum(v.values()) for k, v in inverted_index.items()}

    query_likelihood_models = [
        ### Laplace smoothing
        (
            'Laplace smoothing',
            lidstone_correction,
            {'eps': 1.0},
            LAPLACE_OUTPUT,
        ),
        ### Lidstone correction
        (
            'Lidstone correction',
            lidstone_correction,
            {'eps': 0.1},
            LIDSTONE_OUTPUT,
        ),
        ### Dirichlet smoothing
        (
            'Dirichlet smoothing',
            dirichlet_smoothing,
            {'mu': 50, 'total': total, 'term_freq_sum': term_freq_sum},
            DIRICHLET_OUTPUT,
        )
    ]

    for name, model, kwargs, output_file in query_likelihood_models:
        print(f'Exporting {name} query results to {output_file}...')
        with open(output_file, 'w') as f:
            start_line = ''
            for qid in tqdm(qid_order):
                query_tf = query_sparse_tf[qid]
                pids = qid2candidates[qid]
                scores = [model(query_tf, pid, inverted_index, passage_lengths[pid], **kwargs)
                          for pid in pids]
                top100 = argsort(scores, reverse=True)[:100]  # Descending order
                for arg in top100:
                    f.write(start_line)
                    f.write(f'{qid},{pids[arg]},{scores[arg]}')
                    start_line = '\n'


if __name__ == '__main__':
    task4()
