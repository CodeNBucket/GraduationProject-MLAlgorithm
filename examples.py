file_path = r"C:\Users\Turgut\Desktop\Dataset\output_review_yelpHotelData_NRYRcleaned.txt"
text_file=open(file_path)
reviews=text_file.readlines()#reviews contains all the text information
text_file.close()
file_path=r"C:\Users\Turgut\Desktop\Dataset\output_meta_yelpHotelData_NRYRcleaned.txt"
meta_file=open(file_path)
meta=meta_file.readlines()
labels=[]
for label in meta:
    labels.append(label.split()[4])
print(labels)
X_train,X_test, y_train, y_test = train_test_split( reviews, labels, random_state=0)
print(y_test)


tokenized_docs =[]

"""
tokens = []
            line=line.lower()
            line=re.sub(r'[^a-zA-Z0-9]', ' ', line)
            line = line.split()
            for word in line:
                tokens.append(word)
            t = [token for token in tokens if token not in stopwords.words("english")]
            line = ' '.join(t)
"""

"""for doc in reviews:
    doc=doc.lower()
    doc=doc.split()
    for word in doc:
        tokenized_docs.append(word)
"""

"""example_vector= model.infer_vector(['This', 'hotel', 'is', 'just', 'lame'])
example_vector = np.array(example_vector).reshape(1, -1)
prediction=LogisticRegressionModel.predict(example_vector)
print(prediction)
"""

"""
C_values=[0.001,0.01,0.1,1,10,100,1000]
for C in C_values:
    LogisticRegressionModel = LogisticRegression(class_weight={'N': 1, 'Y': 6},C=C)
    scores=cross_val_score(LogisticRegressionModel,vectors,labels)
    print(C)
    print("Cross-validated scores:", scores)
    print("Average accuracy:", scores.mean())
"""

"""
y_pred=LogisticRegressionModel.predict(X_test)
print(y_pred[10:30])
print(y_pred[240:270])
print("Test set score: {:.2f}".format(LogisticRegressionModel.score(X_test, y_test)))
"""