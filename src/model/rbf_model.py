import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_sklearn_rbf(X_train, y_train, gamma=1.0, n_components=100, random_state=42):

    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
    clf = Pipeline([
        ("rbf_feature", rbf_feature),
        ("linear", Ridge())
    ])
    clf.fit(X_train, y_train)
    return clf

def evaluate_sklearn_rbf(model, X_test, y_test, threshold=0.5):

    y_pred = model.predict(X_test)
    y_pred_class = np.where(y_pred >= threshold, 1, 0)

    print("Genauigkeit:", accuracy_score(y_test, y_pred_class))
    print("Klassifikationsbericht:\n", classification_report(y_test, y_pred_class))
