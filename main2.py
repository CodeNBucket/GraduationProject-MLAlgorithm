from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import gensim
import re
import nltk
from nltk.corpus import stopwords
file_path = r"C:\Users\Turgut\Desktop\Dataset\output_review_yelpHotelData_NRYRcleaned.txt"
text_file=open(file_path)
reviews=text_file.readlines()#reviews contains all the text information
text_file.close()


def read_corpus(f): #Taggs the documents

        for i, line in enumerate(f): #i is index line is the review


            tokens = gensim.utils.simple_preprocess(line) # tokenizes the text
            yield TaggedDocument(tokens, [i]) #yields the tagged document one-by-one to the train corpus in a list format

train_corpus = list(read_corpus(reviews)) #train_corpus contains the list of tokenized tagged documents, it will be used to test the accuracy of the doc2vec process
model = Doc2Vec(vector_size=384, min_count=1, epochs=10,window=10,dm=1)
#vector size:dimensionality of the vecto
# min_count:ignores the words with total frequency lower than the value
# epochs:number of iterations during training
# window:maximum distance between the current and predicted word within a sentence
# dm:defines the training algorithm if dm=1, ‘distributed memory’ (PV-DM) is used
model.build_vocab(train_corpus) #builds the models vocabulary with train_corpus
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs) #trains the model
model.save(r"C:\Users\Turgut\Desktop\Dataset\trained_model")#saves the model


ranks = []
for doc_id in range(model.corpus_count):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words) #takes the inferred vectors one by one
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv)) #compares it to all other documents and list the docs most similar to least similar (doc_id,similarityscore)
    rank = [docid for docid, sim in sims].index(doc_id)# finds the location of doc_id and stores it in the rank(if the inferred_vector is the most similar doc_id will be at the 1'st place(index[0]) which means it is working well)
    ranks.append(rank)#stores them in rank


dict={}
for i in ranks:
    if i==0:
        if 0 in dict.keys():
            dict[0]+=1
        else:
            dict[0]=1
    else:
        if 1 in dict.keys():
            dict[1] += 1
        else:
            dict[1] = 1
