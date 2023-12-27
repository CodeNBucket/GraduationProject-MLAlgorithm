
""" Iris example"""
from sklearn.datasets import load_iris
import numpy as np
iris_dataset = load_iris() #iris_dataset is the dataset
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys())) #keys of dataset
#dict_keys(['target_names', 'feature_names', 'DESCR', 'data', 'target'])
#output names[fake,not fake]   column names    descr   [1,2,3,4]  labels
print("Shape of data: {}".format(iris_dataset['data'].shape)) # how many rows and columns in data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0) #splitting the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new) #X_new is eighter 0,1,2 which represents different targets/labels
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
y_pred = knn.predict(X_test)#similar to 17 we insert dataset to knn.predict which is an algorithm, y_pred is the
# predictions
print("Test set predictions:\n {}".format(y_pred))#shows the predictions
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))#same as print("Test set score: {:.2f}".
# format(knn.score(X_test, y_test)))  #X_train and y_train is used for feeding the algorithm, and then use X_test to see
#the prediction of the algorithm and then compare it with y_test for accuracy
