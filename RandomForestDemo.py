from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

# Load the dataset
data = load_iris()

X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=56, random_state=34,max_depth=2)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
cnf_mtx = confusion_matrix(y_test,y_pred)
print("Accuracy:", accuracy)
print("Confusion matrix:\n",cnf_mtx)


prediction=rf_classifier.predict([[1.5,2.4,3.5,1.1]])
print("prediction for [1.5,2.4,3.5,1.1] : ",prediction,data.target_names[prediction])

