import sys
import spacy
import numpy as np
import pandas as pd
from numba import njit
from numba_progress import ProgressBar
from collections import Counter

ALPHA = 0.1
BETA = 0.1
NUM_TOPICS = 20
MAX_ITER = 200

sp = spacy.load("en_core_web_sm")
np.random.seed(42)

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = pd.DataFrame(lines)[0].sample(frac=1.0, random_state=42).values
    return data

def generate_freq(data, max_docs=6000):
    freqs = Counter()
    all_stopwords = sp.Defaults.stop_words
    all_stopwords.add("enron")
    num_tokens = 0

    for doc in data[:max_docs]:
        tokens = sp.tokenizer(doc)
        for token in tokens:
            token_text = token.text.lower()
            if token_text not in all_stopwords and token.is_alpha:
                num_tokens += 1
                freqs[token_text] += 1

    return freqs

def get_vocab(freqs, freq_threshold=3):
    vocab = {}
    vocab_idx_str = {}
    vocab_idx = 0

    for word in freqs:
        if freqs[word] >= freq_threshold:
            vocab[word] = vocab_idx
            vocab_idx_str[vocab_idx] = word
            vocab_idx += 1

    return vocab, vocab_idx_str

def tokenize_dataset(data, vocab, max_docs=6000):
    num_tokens = 0
    num_docs = 0
    docs = []

    for doc in data[:max_docs]:
        tokens = sp.tokenizer(doc)
        if len(tokens) > 1:
            doc = []
            for token in tokens:
                token_text = token.text.lower()
                if token_text in vocab:
                    doc.append(token_text)
                    num_tokens += 1
            num_docs += 1
            docs.append(doc)

    corpus = []
    for doc in docs:
        corpus_d = []
        for token in doc:
            corpus_d.append(vocab[token])
        corpus.append(np.asarray(corpus_d))

    return docs, corpus

@njit(nogil=True)
def gibbs_sampling(corpus, num_iter, progress_proxy):
    z = []
    num_docs = len(corpus)

    for _, doc in enumerate(corpus):
        zd = np.random.randint(low=0, high=NUM_TOPICS, size=len(doc))
        z.append(zd)

    nmk = np.zeros((num_docs, NUM_TOPICS))
    for m in range(num_docs):
        for k in range(NUM_TOPICS):
            nmk[m, k] = np.sum(z[m] == k)

    nkw = np.zeros((NUM_TOPICS, vocab_size))
    for doc_idx, doc in enumerate(corpus):
        for i, word in enumerate(doc):
            topic = z[doc_idx][i]
            nkw[topic, word] += 1

    nk = np.sum(nkw, axis=1)
    topic_list = [i for i in range(NUM_TOPICS)]

    for _ in range(num_iter):
        for doc_idx, doc in enumerate(corpus):
            for i in range(len(doc)):
                word = doc[i]
                topic = z[doc_idx][i]

                nmk[doc_idx, topic] -= 1
                nkw[topic, word] -= 1
                nk[topic] -= 1

                # pk = math.gamma(nmk[doc_idx, :] + ALPHA) / math.gamma(ALPHA)

                pz = (nmk[doc_idx, :] + ALPHA) * (nkw[:, word] + BETA) / (nk[:] + BETA*vocab_size)
                cumulative_distribution = np.cumsum(pz)
                cumulative_distribution /= cumulative_distribution[-1]
                topic = np.searchsorted(cumulative_distribution, topic_list, side="right")[0]

                z[doc_idx][i] = topic
                nmk[doc_idx, topic] += 1
                nkw[topic, word] += 1
                nk[topic] += 1
        progress_proxy.update(1)

    return z, nmk, nkw, nk

data = read_file(sys.argv[1])
freqs = generate_freq(data)
vocab, vocab_idx_str = get_vocab(freqs)
docs, corpus = tokenize_dataset(data, vocab)
vocab_size = len(vocab)

with ProgressBar(total=MAX_ITER) as progress:
    z, nmk, nkw, nk = gibbs_sampling(corpus, MAX_ITER, progress)

# phi = nkw / nk.reshape(NUM_TOPICS, 1)
# num_words = 10
# for k in range(NUM_TOPICS):
#     most_common_words = np.argsort(phi[k])[::-1][:num_words]
#     print(f"Topic {k} most common words: ")
#     for word in most_common_words:
#         print(vocab_idx_str[word])
#     print("\n")
#
