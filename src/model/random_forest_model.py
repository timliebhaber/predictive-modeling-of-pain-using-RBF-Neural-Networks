from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Accuracy RF: ", accuracy_score(y_test, y_pred))
    print("Classification Report RF:\n", classification_report(y_test, y_pred))
    
    return model