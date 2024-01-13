from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

file_path = r"data/textdata.txt"
text_file = open(file_path)
reviews = text_file.readlines()  # reviews contains all the text information
text_file.close()

def read_corpus(f):  # Taggs the documents
    lemmatizer = WordNetLemmatizer()
    for i, line in enumerate(f):  # i is index line is the review

        tokens = []

        line = line.lower()  # converting the lines to lowercase

        line = re.sub(r'[^a-zA-Z0-9\s]', '', line)  # removes non-alphanumeric characters

        words = line.split()  # splits the line into words

        words = [lemmatizer.lemmatize(word) for word in
                 words]  # applies lemmitization which is a process of reducing the words to its base form running-run

        t = [token for token in words if
             len(token) > 2 and not token.isnumeric() and token not in stopwords.words("english")]

        # removes words that their lenght is shorter than 2
        # removes words that are numeric values
        # removes common words which is in the list of stopwords('stop','the','and','that')

        yield TaggedDocument(t, [i])  # yields the tagged document one-by-one to the train corpus in a list format


train_corpus = list(read_corpus(
    reviews))  # train_corpus contains the list of tokenized tagged documents, it will be used to test the accuracy of the doc2vec process
model = Doc2Vec(vector_size=128, min_count=1, epochs=10, window=10, dm=1)
model.wv.save_word2vec_format('models/doc2vec.wordvectors',binary=True)
# vector size:dimensionality of the vector
# min_count:ignores the words with total frequency lower than the value
# epochs:number of iterations during training
# window:maximum distance between the current and predicted word within a sentence
# dm:defines the training algorithm if dm=1, ‘distributed memory’ (PV-DM) is used, it is a prediction based model and uses backpropagation for tuning its parameters
model.build_vocab(train_corpus)  # builds the models vocabulary with train_corpus
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)  # trains the model
model.save(r"models/doc2vec_model")  # saves the model

ranks = []
for doc_id in range(model.corpus_count):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)  # takes the inferred vectors one by one
    sims = model.dv.most_similar([inferred_vector], topn=len(
        model.dv))  # compares it to all other documents and list the docs most similar to least similar (doc_id,similarityscore)
    rank = [docid for docid, sim in sims].index(
        doc_id)  # finds the location of doc_id and stores it in the rank(if the inferred_vector is the most similar doc_id will be at the 1'st place(index[0]) which means it is working well)
    ranks.append(rank)  # stores them in rank

dict = {}
for i in ranks:
    if i == 0:
        if 0 in dict.keys():
            dict[0] += 1
        else:
            dict[0] = 1
    else:
        if 1 in dict.keys():
            dict[1] += 1
        else:
            dict[1] = 1

accuracy_percentage = (dict[0] / (dict[0] + dict[1])) * 100
formatted_accuracy = "{:.2f}".format(accuracy_percentage)
print("While there are 5854 reviews used for training the model, while testing the models vectorization accuracy, "
      "of the vectors, " + str(dict[0]) + " have been found most similar with itself. "
                                          "When the trained model and inferred vectors created afterwards are compared, "
                                          "the Doc2Vec has %" + formatted_accuracy + " accuracy which is accaptable")
