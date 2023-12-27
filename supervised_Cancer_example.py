
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)
svm = SVC(C=100)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() #for data processing/scaling the inputs between 0-1
scaler.fit(X_train) #for getting X_train informations, stores it at scaler
X_train_scaled = scaler.transform(X_train) #does the transformation operation
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print("Scaled test set accuracy: {:.2f}".format(
svm.score(X_test_scaled, y_test)))













