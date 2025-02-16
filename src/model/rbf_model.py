import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def train_and_test_sklearn_rbf(
    X_train,
    y_train,
    X_test,
    y_test,
    gamma,
    n_components,
    random_state,
    threshold=0.5
):
    """
    Kombinierte Funktion die folgendes durchführt:
    1. Trainiert ein RBF-Ridge Modell mit RBFSampler und Ridge Regression
    2. Evaluierung durch Vorhersagen auf Testdaten
    3. Klassifizierung mittels Schwellenwert
    4. Gibt Modell, Scores und klassifizierte Vorhersagen zurück
    
    Parameter:
    X_train, y_train - Trainingsdaten
    X_test, y_test - Testdaten und Labels
    gamma - RBF Kernel Parameter
    n_components - Anzahl der RBF-Komponenten
    random_state - Seed für Reproduzierbarkeit
    threshold - Klassifizierungsschwelle (0-1)
    
    Returns:
    Tuple (Modell, Vorhersage-Scores, klassifizierte Vorhersagen)
    """
    
    model = Pipeline([
    ("scaler", StandardScaler()),  # Add scaling FIRST (crucial for RBF)
    ("rbf_feature", RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)),
    ("linear", Ridge())
])
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_scores = model.predict(X_test)
    y_pred_class = np.where(y_scores >= threshold, 1, 0)
    
    # Metriken ausgeben
    print("Accuracy RBF:", accuracy_score(y_test, y_pred_class))
    print("Classification report RBF:\n", classification_report(y_test, y_pred_class))
    
    return model, y_scores, y_pred_class