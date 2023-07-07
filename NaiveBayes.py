from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd

# iris = load_iris()
# X = iris.data 
# y = iris.target

df = pd.read_csv("Iris.csv")

X = df.iloc[: ,1:5]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

classifier = GaussianNB()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

print("\naccuracy score: ",accuracy_score(y_test,y_pred))
print("\nconfusion matrix:\n",confusion_matrix(y_test,y_pred))

# from sklearn import datasets
# from sklearn import metrics
# from sklearn.naive_bayes import GaussianNB
# # load the iris datasets
# dataset = datasets.load_iris()
# # fit a Naive Bayes model to the data
# model = GaussianNB()

# model.fit(dataset.data, dataset.target)
# print(model)
# # make predictions
# expected = dataset.target
# predicted = model.predict(dataset.data)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))