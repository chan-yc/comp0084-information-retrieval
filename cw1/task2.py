import os
import pickle

from tqdm import tqdm
from task1 import ASSETS_PATH, CANDIDATE_FILE, Tokeniser, load_index, load_data


# File paths
INVERTED_INDEX = os.path.join(ASSETS_PATH, '_task2_inverted_index.pkl')
PASSAGE_LENGTHS = os.path.join(ASSETS_PATH, '_task2_passage_lengths.pkl')


def load_inverted_index():
    """For use in later tasks"""
    return load_data(binary=INVERTED_INDEX,
                     fallback_func=lambda: get_inverted_index(load_index()))


def load_passage_lengths():
    """For use in later tasks"""
    return load_data(binary=PASSAGE_LENGTHS,
                     fallback_func=lambda: get_passage_lengths(load_index()))


def get_inverted_index(index):
    inverted_index = {}  # {term: {pid: count}}
    pids = set()
    print('Computing the inverted index...')
    with open(CANDIDATE_FILE, 'r') as f:
        for record in tqdm(f.readlines()):
            _, pid, _, passage = record.split('\t')
            if pid in pids:
                continue
            pids.add(pid)
            for term in index[passage]:
                # Count term occurence in passage
                if term not in inverted_index:
                    inverted_index[term] = {}
                    inverted_index[term][pid] = 1
                elif pid not in inverted_index[term]:
                    inverted_index[term][pid] = 1
                else:
                    inverted_index[term][pid] += 1

    # Dump the inverted index for later tasks
    with open(INVERTED_INDEX, 'wb+') as f1:
        pickle.dump(inverted_index, f1)
    print(f'Inverted index saved to {INVERTED_INDEX}')
    print(len(inverted_index))
    return inverted_index


def get_passage_lengths(index):
    passage_lengths = {}  # {pid: number_of_terms}
    print('Computing the passages lengths...')
    with open(CANDIDATE_FILE, 'r') as f:
        for record in tqdm(f.readlines()):
            _, pid, _, passage = record.split('\t')
            if pid in passage_lengths:
                continue
            passage_lengths[pid] = len(index[passage])

    # Dump the passage lengths for later tasks
    with open(PASSAGE_LENGTHS, 'wb+') as f1:
        pickle.dump(passage_lengths, f1)
    print(f'Passage lengths saved to {PASSAGE_LENGTHS}')

    return passage_lengths


def remove_stopwords_from_index(index):
    """Remove stopwords from the index in place."""
    print('Removing stopwords from the index...')
    for passage in tqdm(index):
        tokens = index[passage]
        index[passage] = [t for t in tokens if not Tokeniser.is_stop(t)]


def task2():
    """Task 2 of the coursework 1."""
    print('-'*50 + '\nTask 2\n' + '-'*50)

    # Get the index without stopwords
    index = load_index()  # stopwords included
    remove_stopwords_from_index(index)

    get_inverted_index(index)
    get_passage_lengths(index)


if __name__ == '__main__':
    task2()
