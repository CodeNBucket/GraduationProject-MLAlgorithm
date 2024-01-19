import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import defaultdict


def compute_probabilities(graph, probs, p, q):
    G = graph
    for source_node in G.nodes():
        for current_node in G.neighbors(source_node):
            probs_ = list()
            for destination in G.neighbors(current_node):
                if source_node == destination:
                    prob_ = G[current_node][destination].get('weight', 1) * (1/p)
                elif destination in G.neighbors(source_node):
                    prob_ = G[current_node][destination].get('weight', 1)
                else:
                    prob_ = G[current_node][destination].get('weight', 1) * (1/q)

                probs_.append(prob_)

            probs[source_node]['probabilities'][current_node] = probs_/np.sum(probs_)

    return probs


def generate_random_walks(graph, probs, max_walks, walk_len):
    G = graph
    walks = list()
    for start_node in G.nodes():
        for i in range(max_walks):
            walk = [start_node]
            walk_options = list(G[start_node])
            if len(walk_options) == 0:
                break
            first_step = np.random.choice(walk_options)
            walk.append(first_step)

            for k in range(walk_len - 2):
                walk_options = list(G[walk[-1]])
                if len(walk_options) == 0:
                    break
                probabilities = probs[walk[-2]]['probabilities'][walk[-1]]
                next_step = np.random.choice(walk_options, p=probabilities)
                walk.append(next_step)

            walks.append(walk)

    np.random.shuffle(walks)
    walks = [list(map(str, walk)) for walk in walks]

    return walks


def Node2Vec(generated_walks, window_size, embedding_vector_size):
    model = Word2Vec(
        sentences=generated_walks,
        window=window_size,
        vector_size=embedding_vector_size,
    )

    return model
