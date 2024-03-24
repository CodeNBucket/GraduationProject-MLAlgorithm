import gensim
import pandas as pd
import smart_open

metadata = pd.read_csv("../data/metadata.txt", header=None, sep="\\s+")

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


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(
                    tokens, [metadata["ReviewId"][i]]
                )


corpus = list(read_corpus("../data/textdata.txt"))

model = gensim.models.doc2vec.Doc2Vec(
    vector_size=384, min_count=1, window=10, epochs=10, dm=1
)

model.build_vocab(corpus)

model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.dv.save_word2vec_format("../models/doc2vec.wordvectors", binary=True)

model.save(
    "../models/doc2vec.model",
)

