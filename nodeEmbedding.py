import pandas as pd
import networkx as nx
from node2vec import Node2Vec

metadata = pd.read_csv(
    'data/metadata.txt',
    header=None,
    delim_whitespace=True
)

metadata.columns = [
    'Date', 'ReviewId', 'ReviewerId',
    'ProductId', 'Label', 'Useful',
    'Funny', 'Cool', 'Rating'
]

G = nx.Graph()

G.add_weighted_edges_from(
    metadata[['ReviewerId', 'ProductId', 'Rating']].values
)

node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=10,
    num_walks=10,
    workers=4
)

model = node2vec.fit(
    window=10
)

model.wv.save_word2vec_format('models/node2vec.wordvectors', binary=True)
