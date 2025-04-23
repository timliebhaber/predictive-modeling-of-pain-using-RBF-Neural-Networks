from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def train_svm(
        X, y, groups,
        *,
        kernel="rbf",        
        C=1.0,               
        gamma="scale",       
        random_state=20
    ):

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    )

    loo = LeaveOneGroupOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred.extend(model.predict(X_test))
        y_true.extend(y_test)

    print("Accuracy SVM:", accuracy_score(y_true, y_pred))
    print("Classification Report SVM:\n",
          classification_report(y_true, y_pred))

    model.fit(X, y)
    return model
