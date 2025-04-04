from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut


def train_random_forest(X, y):  # Updated function signature
    model = RandomForestClassifier(n_estimators=99, random_state=42)
    
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_true.append(y_test[0])
        y_pred.append(pred[0])
    
    print("Accuracy RF (LOO): ", accuracy_score(y_true, y_pred))
    print("Classification Report RF (LOO):\n", classification_report(y_true, y_pred))
    
    # Train final model on entire dataset
    model.fit(X, y)
    return model