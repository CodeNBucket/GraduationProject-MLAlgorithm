import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
reviews_train = pd.read_csv(r"C:\Users\Turgut\Desktop\Dataset\output_review_yelpHotelData_NRYRcleaned.txt",on_bad_lines='skip')
reviews_label=open(r"C:\Users\Turgut\Desktop\Dataset\output_meta_yelpHotelData_NRYRcleaned.txt")
label=[]
for revies in reviews_label:
    label.append(revies.split()[4])
print(label)
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train,label #labels and data
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[1]:\n{}".format(text_train))# text_train list contains all the reviews
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))
vect = CountVectorizer()# now vect has all the informations about the reviews including an array of all the words
vect.fit(text_train)
X_train = vect.transform(text_train) #now x_train contains the vectors of each sample
print("X_train:\n{}".format(repr(X_train)))
