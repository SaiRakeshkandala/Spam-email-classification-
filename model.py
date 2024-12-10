from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
            model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                    print("Accuracy:", accuracy_score(y_test, y_pred))
                        print("Classification Report:\n", classification_report(y_test, y_pred))
                            return model
                            