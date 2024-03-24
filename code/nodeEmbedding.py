import networkx as nx
import pandas as pd

from node2vec import *

metadata = pd.read_csv("../data/metadata.txt", header=None, delim_whitespace=True)

metadata.columns = [
    "Date",
    "ReviewId",
    "ReviewerId",
    "ProductId",
    "Label",
    "Useful",
    "Funny",
    "Cool",
    "Rating",
]

G = nx.Graph()

G.add_weighted_edges_from(metadata[["ReviewerId", "ProductId", "Rating"]].values)

probs = defaultdict(dict)

for node in G.nodes():
    probs[node]["probabilities"] = dict()

cp = compute_probabilities(G, probs, 1, 1)

walks = generate_random_walks(G, cp, 10, 10)

model = Node2Vec(walks, 10, 128)

model.wv.save_word2vec_format("../models/node2vec.wordvectors", binary=True)

model.save(
    "../models/node2vec.model",
)

