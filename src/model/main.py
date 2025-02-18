import sys
import os

# Füge `src/` zum Python-Suchpfad hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.model.mlp_model import train_mlp
from src.model.random_forest_model import train_random_forest
from src.model.rbf_model import train_and_test_sklearn_rbf

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


def main():
    # Schritt 1: Generiere Trainings- und Testdaten
    X_train, X_test, y_train, y_test = generate_data(data_dir="data/combined")
    
    # Step 2: Train and evaluate the RBFN model
    model_rbf = train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=200, random_state=42)

    # Step 2b: Train and evaluate Random Forest model
    model_rf = train_random_forest(X_train, X_test, y_train, y_test)

    # Step 2c: Train and evaluate MLP model
    model_mlp = train_mlp(X_train, X_test, y_train, y_test)
    
    # Schritt 3: Visualisiere die Ergebnisse
    """
    plot_confusion_matrix(y_test, y_pred_class)
    plot_roc_curve(y_test, y_scores)
    plot_prediction_distribution(y_test, y_scores)
    """
if __name__ == "__main__":
    main()
