import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_sklearn_rbf(X_train, y_train, gamma=1.0, n_components=100, random_state=42):
    """
    Trainiert ein Modell, das die Eingabedaten zunächst mithilfe eines RBFSamplers in einen 
    höherdimensionalen Raum abbildet und anschließend ein Ridge-Regressionsmodell verwendet.
    """
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)
    clf = Pipeline([
        ("rbf_feature", rbf_feature),
        ("linear", Ridge())
    ])
    clf.fit(X_train, y_train)
    return clf

def evaluate_sklearn_rbf(model, X_test, y_test, threshold=0.5):
    """
    Evaluiert das Modell: Es werden Vorhersagen generiert und mittels eines Schwellenwerts in Klassen
    (0 oder 1) umgewandelt. Anschließend werden Genauigkeit und ein detaillierter Klassifikationsbericht ausgegeben.
    """
    y_scores = model.predict(X_test)
    y_pred_class = np.where(y_scores >= threshold, 1, 0)
    
    print("Accuracy RBF:", accuracy_score(y_test, y_pred_class))
    print("Classification report RBF:\n", classification_report(y_test, y_pred_class))
    
    return y_scores, y_pred_class

