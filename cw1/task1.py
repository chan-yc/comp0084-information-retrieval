import os
import re
import math
import pickle
import gensim
import matplotlib.pyplot as plt

from tqdm import tqdm


# File paths
DATA_PATH = os.path.abspath('./data')
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.abspath('.')  # The coursework expects data in the same directory

QUERY_FILE = os.path.join(DATA_PATH, 'test-queries.tsv')
PASSAGE_FILE = os.path.join(DATA_PATH, 'passage-collection.txt')
CANDIDATE_FILE = os.path.join(DATA_PATH, 'candidate-passages-top1000.tsv')

ASSETS_PATH = os.path.abspath('./assets')
if not os.path.exists(ASSETS_PATH):
    os.makedirs(ASSETS_PATH)
FIGURE_1 = os.path.join(ASSETS_PATH, '_task1_fig1.pdf')
FIGURE_2 = os.path.join(ASSETS_PATH, '_task1_fig2.pdf')
FIGURE_3 = os.path.join(ASSETS_PATH, '_task1_fig3.pdf')
INDEX = os.path.join(ASSETS_PATH, '_task1_index.pkl')


# Plots settings
plt.rcParams['font.size'] = 12
plt.rcParams["figure.autolayout"] = True


# English stopword list from NLTK
# See https://www.nltk.org/howto/corpus.html?highlight=stopwords#word-lists-and-lexicons
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


class Tokeniser:

    def __init__(self):
        self._preprocesses = [
            self._replace_non_alphanumeric,
            self._split_alphanumeric,
            gensim.parsing.preprocessing.stem_text,
        ]

    def __call__(self, doc):
        for process in self._preprocesses:
            doc = process(doc)
        return self._split_by_space(doc)

    @staticmethod
    def is_stop(token):
        """Stopword defined by NLTK"""
        return (token in STOPWORDS)

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


def compute_index():
    """Get the index of the passage collection."""
    index = {}  # {passage_full_text: [tokens]}
    tokeniser = Tokeniser()

    print('Computing the index...')
    with open(PASSAGE_FILE, 'r') as f:
        for passage in tqdm(f.readlines()):
            # All tokens, including stopwords
            index[passage] = tokeniser(passage)

    # Dump the index for later tasks
    with open(INDEX, 'wb+') as f1:
        pickle.dump(index, f1)
    print(f'Index saved to {INDEX}')

    return index


def zipf_law_constant(N):
    """Calculate the constant for Zipf's Law."""
    return 1 / sum([1/r for r in range(1, N+1)])


def normalise_freq(counts):
    """Return descending normalise frequency from counts."""
    total = sum(counts.values())
    return sorted([v / total for v in counts.values()], reverse=True)


def median(values):
    """Return the median of a list of values."""
    values, n = sorted(values), len(values)
    if n % 2 == 0:
        return (values[n//2-1] + values[n//2]) / 2
    else:
        return values[n//2]


def std(values):
    """Return the standard deviation of a list of values."""
    mean = sum(values) / len(values)
    return math.sqrt(sum([(v-mean)**2 for v in values]) / len(values))


def print_counts_stat(counts):
    freqs = normalise_freq(counts)
    ranks = list(range(1, len(freqs)+1))
    freq_rank = [f*r for f, r in zip(freqs, ranks)]
    print(f'Vocabulary size: {len(counts)}')
    print(f"Theoretical Zipf's law constant: {zipf_law_constant(len(counts)):.6f}")
    print(f'Empirical Median of freq * rank: {median(freq_rank):.6f}')
    print(f'Empirical Mean of freq * rank  : {sum(freq_rank)/len(freq_rank):.6f}')
    print(f'Empirical Stdv of freq * rank  : {std(freq_rank):.6f}')


def frequency_rank_plot(counts, file, log=False):
    """Plot probability of occurence against rank with Zipf's Law."""
    probs = normalise_freq(counts)
    ranks = list(range(1, len(probs)+1))
    C = zipf_law_constant(len(probs))
    zipf_probs = [C / r for r in ranks]  # Zipf's Law

    plt.figure(figsize=(5, 4))
    plt.plot(ranks, probs, label='Normalised Frequency')
    plt.plot(ranks, zipf_probs, ':r', label="Zipf's Law")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.legend()
    if log:
        plt.xscale('log')
        plt.yscale('log')
    plt.savefig(file)
    print(f'Figure saved to {file}')


def load_index():
    """For use in later tasks"""
    return load_data(binary=INDEX, fallback_func=compute_index)


def load_data(binary, fallback_func):
    """Load data from file or fallback to a function."""
    try:
        with open(binary, 'rb') as f:
            return pickle.load(f)
    except:
        return fallback_func()


def task1():
    """Task 1 of the coursework 1."""
    print('-'*50 + '\nTask 1\n' + '-'*50)

    ### Indexing
    index = compute_index()
    all_counts = {}           # {token: counts}
    non_stopword_counts = {}  # {token: counts}

    print('Counting the frequency...')
    for _, tokens in tqdm(index.items()):
        for token in tokens:
            if token not in all_counts:
                all_counts[token] = 1
            else:
                all_counts[token] += 1
            if not Tokeniser.is_stop(token):
                if token not in non_stopword_counts:
                    non_stopword_counts[token] = 1
                else:
                    non_stopword_counts[token] += 1

    ### Stats for report
    print('### Distribution of tokens (stopwords INCLUDED) ###')
    print_counts_stat(all_counts)
    print()
    print('### Distribution of tokens (stopwords EXCLUDED) ###')
    print_counts_stat(non_stopword_counts)
    print()

    ### Figure 1
    # Normalised frequency against rank
    frequency_rank_plot(all_counts, FIGURE_1)

    ### Figure 2
    # Normalised frequency against rank (log-log)
    frequency_rank_plot(all_counts, FIGURE_2, log=True)

    ### Figure 3
    # Normalised frequency against rank after removing stopwords (log-log)
    frequency_rank_plot(non_stopword_counts, FIGURE_3, log=True)


if __name__ == '__main__':
    task1()
