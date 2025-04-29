import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def train_svm_LOGO(
        X, y, groups,
        *,
        kernel="rbf",        
        C=1.0,               
        gamma="scale"
    ):

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma)
    )

    loo = LeaveOneGroupOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred.extend(model.predict(X_test))
        y_true.extend(y_test)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred,  average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred,     average="weighted", zero_division=0)


    print("Accuracy SVM:", accuracy_score(y_true, y_pred))
    print("Classification Report SVM:\n",
          classification_report(y_true, y_pred))
    
    df_row = pd.DataFrame([{
        "precision": prec,
        "recall"   : rec,
        "accuracy" : acc,
        "f1_score" : f1
    }])

    csv_path = "svm_logo_run_metrics.csv"
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True,  index=False)


    model.fit(X, y)
    return model

def train_svm(
        X_train, X_test, y_train, y_test,
        *,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        refit_full=False
    ):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred,  average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred,     average="weighted", zero_division=0)

    # Evaluation auf Testdaten
    y_pred = model.predict(X_test)
    print("Accuracy SVM:", accuracy_score(y_test, y_pred))
    print("Classification Report SVM:\n",
          classification_report(y_test, y_pred))
    
    df_row = pd.DataFrame([{
        "precision": prec,
        "recall"   : rec,
        "accuracy" : acc,
        "f1_score" : f1
    }])

    csv_path = "svm_run_metrics.csv"
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True,  index=False)
    
    return model
