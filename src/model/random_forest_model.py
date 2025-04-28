from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut


def train_random_forest_LOGO(X, y, groups):
    model = RandomForestClassifier(n_estimators=99, random_state=20)
    
    loo = LeaveOneGroupOut()
    y_true = []
    y_pred = []
    
    for train_index, test_index in loo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)
    
    print("Accuracy RF: ", accuracy_score(y_true, y_pred))
    print("Classification Report RF:\n", classification_report(y_true, y_pred))
    
    model.fit(X, y)
    return model

def train_random_forest(
        X_train, X_test, y_train, y_test,
        *,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=20,
        refit_full=False
    ):
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    # Training
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("Accuracy RF:", accuracy_score(y_test, y_pred))
    print("Classification Report RF:\n",
          classification_report(y_test, y_pred))
    return model