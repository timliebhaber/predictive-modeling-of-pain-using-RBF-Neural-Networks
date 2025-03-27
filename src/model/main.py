import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.mlp_model import train_mlp
from src.model.random_forest_model import train_random_forest
from src.model.rbf_model import train_and_test_sklearn_rbf

def generate_data(data_dir="data/combined"):

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
            
            df = pd.read_csv(file_path)
            
            # Extrahiere statistische Features aus der gesamten Zeitreihe
            features = {
                "gsr_mean": df["gsr"].mean(),
                "gsr_std": df["gsr"].std(),
                "ecg_max": df["ecg"].max(),
                "emg_energy": (df["emg_trapezius"] ** 2).sum(), # Quadriert um nur positive Werte zu erhalten
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
    print("\nGenerating data.\n")
    X_train, X_test, y_train, y_test = generate_data(data_dir="data/combined")
    
    # Schritt 2: Trainiere und evaluiere Modelle
    print("\nTraining and evaluating models.\n\n\n")
    model_rbf = train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=200, random_state=42)

    model_rf = train_random_forest(X_train, X_test, y_train, y_test)

    model_mlp = train_mlp(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
