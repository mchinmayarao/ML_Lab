from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

documents = [
    ("I love this movie", "pos"),
    ("This was an awesome movie", "pos"),
    ("I disliked this movie", "neg"),
    ("This movie was terrible", "neg"),
    ("I enjoyed watching the movie", "pos"),
    ("The movie was awfull","neg"),
    ("I liked the movie","pos")
]

text, labels = zip(*documents)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)
clf = MultinomialNB()
clf.fit(X, labels)

test_documents = [
    "I really did not like this movie",
    "This movie was  awful"
]

x_test = vectorizer.transform(test_documents)
pred_labels = clf.predict(x_test)

for doc, label in zip(test_documents, pred_labels):
    print(f"{doc} ----> {label}")

# Compute accuracy
accuracy = accuracy_score(["neg", "neg"], pred_labels)
print("Accuracy:", accuracy)

# Compute precision
precision = precision_score(["neg", "neg"], pred_labels, pos_label="neg")
print("Precision:", precision)

# Compute recall
recall = recall_score(["neg", "neg"], pred_labels, pos_label="neg")
print("Recall:", recall)
