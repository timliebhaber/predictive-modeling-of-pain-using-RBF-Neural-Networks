from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from sklearn.metrics import precision_score, recall_score, f1_score 
import pandas as pd

def train_mlp(X_train, X_test, y_train, y_test):
    #StandardScaler standardisiert die Daten, damit der Mittelwert 0 und die Standardabweichung 1 ist
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam')
    
    mlp.fit(X_train_scaled, y_train)
    
    y_pred = mlp.predict(X_test_scaled)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred)

    run_df = pd.DataFrame([{
        "precision": prec,
        "recall"   : rec,
        "accuracy" : acc,
        "f1_score" : f1
    }])

    csv_path = "mlp_run_metrics.csv" 

    if os.path.exists(csv_path):
        run_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        run_df.to_csv(csv_path, mode="w", header=True,  index=False)
    
    print("Accuracy MLP: ", accuracy_score(y_test, y_pred))
    print("Classification Report MLP:\n", classification_report(y_test, y_pred))
    
    return mlp