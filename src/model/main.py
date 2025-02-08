import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def generate_data(data_dir="data/combined"):
 
    X = []
    y = []
    
    # Durchlaufe alle Dateien im Verzeichnis
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            
            if "-BL1-" in file_name:
                label = 0  # Kein Schmerz
            elif "-PA4-" in file_name:
                label = 1  # Starker Schmerz
            else:
                continue 
            
            df = pd.read_csv(file_path)
            
            # Extrahiere die relevanten Features
            features = df[["time", "gsr", "ecg", "emg_trapezius", "temp_adj"]].values
            
            # Füge die Zeilen (Features) und das jeweilige Label hinzu
            X.extend(features)
            y.extend([label] * len(features))
    
    # Konvertiere die Arrays zu float32
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Aufteilung in Trainings- und Testdaten (80/20) unter Erhaltung der Klassenverteilung
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True,
        stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Labels: {y_test.shape}")
    print("\nClass distribution:")
    print(f"Training: {np.unique(y_train, return_counts=True)}")
    print(f"Test: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_test, y_train, y_test

def train_sklearn_rbf(X_train, y_train, gamma=1.0, n_components=100, random_state=42):

   # Modell,wird mit RBFSampler trainiert und die Daten in einen werden  in höherdimensionalen Raum abgebildet und anschließend wird ein Ridge-Regressionsmodell verwendet.

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

def main():
    # Schritt 1: Generiere Trainings- und Testdaten
    X_train, X_test, y_train, y_test = generate_data(data_dir="data/combined")
    
    # Schritt 2: Trainiere das RBFN-Modell
    model = train_sklearn_rbf(X_train, y_train, gamma=1.0, n_components=100, random_state=42)
    
    # Schritt 3: Bewerte das trainierte Modell
    evaluate_sklearn_rbf(model, X_test, y_test, threshold=0.5)

if __name__ == "__main__":
    main()