import pandas as pd
import networkx as nx

metadata = pd.read_csv('data/metadata.txt', header=None, delim_whitespace=True)
metadata.columns = ['Date', 'ReviewId', 'ReviewerId', 'ProductId', 'Label', 'Useful', 'Funny', 'Cool', 'Rating']

G = nx.Graph()

for i in metadata.index:
    G.add_node(metadata['ReviewerId'][i], type='reviewer')
    G.add_node(metadata['ProductId'][i], type='product')
    G.add_edge(metadata['ReviewerId'][i], metadata['ProductId'][i], weight=int(metadata['Rating'][i]))

# print(G.edges[("_h5MGqTBM8J0tuAfbEvVDg", "33Xc1Bk_gkSY5xb2doQ7Ng")])
