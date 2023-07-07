from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#print(len(X_train, X_test, y_train, y_test))

# def plot_dataset(X, y, axes):
#     plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
#     plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
#     plt.axis(axes)
#     plt.grid(True, which='both')
#     plt.xlabel(r"$x_1$", fontsize=20)
#     plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

# plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
# plt.show()

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X_train, y_train)

y_pred = polynomial_svm_clf.predict(X_test)

#print("\npredictions\n",polynomial_svm_clf.predict(X_test))
print("\naccuracy score: ",accuracy_score(y_pred,y_test))
print("\nconfusion matrix:\n",confusion_matrix(y_pred,y_test))
