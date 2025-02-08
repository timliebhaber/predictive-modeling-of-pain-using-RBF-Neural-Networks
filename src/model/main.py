import sys
import os

# Füge `src/` zum Python-Suchpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from src.visualize import plot_confusion_matrix, plot_roc_curve, plot_prediction_distribution

def generate_data(data_dir="data/combined"):
    """
    Liest alle CSV-Dateien im Ordner data/combined ein, extrahiert anhand des Dateinamens das Label 
    (-BL1-: Kein Schmerz, -PA4-: Starker Schmerz) und berechnet aus der Zeitreihe statistische Features.
    Verwendete Features:
      - gsr_mean: Mittelwert des GSR-Signals
      - gsr_std: Standardabweichung des GSR-Signals
      - ecg_max: Maximaler ECG-Wert
      - emg_energy: Summe der quadrierten EMG-Werte (Energie des Signals)
    Anschließend wird ein stratified 80/20-Split in Trainings- und Testdaten durchgeführt.
    """
    X = []
    y = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            
            # Label bestimmen
            if "-BL1-" in file_name:
                label = 0  # Kein Schmerz
            elif "-PA4-" in file_name:
                label = 1  # Starker Schmerz
            else:
                continue 
            
            # Lade die gesamte CSV-Datei
            df = pd.read_csv(file_path)
            
            # Extrahiere statistische Features aus der gesamten Zeitreihe
            features = {
                "gsr_mean": df["gsr"].mean(),
                "gsr_std": df["gsr"].std(),
                "ecg_max": df["ecg"].max(),
                "emg_energy": (df["emg_trapezius"] ** 2).sum(),
            }
            
            X.append(features)
            y.append(label)
    
    # Konvertiere in DataFrame zur einfachen Weiterverarbeitung
    X = pd.DataFrame(X)
    y = np.array(y, dtype=np.float32)
    
    # Train-Test-Split (80/20) unter Beibehaltung der Klassenverteilung
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

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
    
    print("Genauigkeit:", accuracy_score(y_test, y_pred_class))
    print("Klassifikationsbericht:\n", classification_report(y_test, y_pred_class))
    
    return y_scores, y_pred_class

def main():
    # Schritt 1: Generiere Trainings- und Testdaten
    X_train, X_test, y_train, y_test = generate_data(data_dir="data/combined")
    
    # Schritt 2: Trainiere das RBFN-Modell
    model = train_sklearn_rbf(X_train, y_train, gamma=1.0, n_components=100, random_state=42)
    
    # Schritt 3: Evaluiere das trainierte Modell
    y_scores, y_pred_class = evaluate_sklearn_rbf(model, X_test, y_test, threshold=0.5)
    
    # Schritt 4: Visualisiere die Ergebnisse
    plot_confusion_matrix(y_test, y_pred_class)
    plot_roc_curve(y_test, y_scores)
    plot_prediction_distribution(y_test, y_scores)

if __name__ == "__main__":
    main()
