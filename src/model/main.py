import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import neurokit2 as nk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.mlp_model import train_mlp
from src.model.random_forest_model import train_random_forest
from src.model.rbf_model import train_and_test_sklearn_rbf
from src.model.svm_model import train_svm

def compute_features(signal_dict):
    features = {}
    
    # ECG Features
    try:
        ecg_signal = signal_dict["ecg"]
        features["ecg_max"] = np.max(ecg_signal)
    except KeyError:
        features["ecg_max"] = 0.0

    # GSR Features
    try:
        gsr_signal = signal_dict["gsr"]
                   
        eda_signals, _ = nk.eda_process(gsr_signal, sampling_rate=1000)
        tonic = eda_signals["EDA_Tonic"]
        
        if tonic.empty:
            raise ValueError("No valid tonic component")
             
        diff1 = np.diff(gsr_signal)
        diff2 = np.diff(gsr_signal, 2)
        
        gsr_features = {
            "gsr_mean": tonic.mean(),
            "gsr_std": tonic.std(),
            "gsr_min": tonic.min(),
            "gsr_max": tonic.max(),
            "gsr_variance": tonic.var(),
            "gsr_skewness": skew(tonic, bias=False),
            "gsr_kurtosis": kurtosis(tonic, bias=False, fisher=False),
            "gsr_abs_first_diff_mean": np.abs(diff1).mean(),
            "gsr_abs_second_diff_mean": np.abs(diff2).mean(),
            "gsr_mean_diff": diff1.mean(),
            "gsr_rms": np.sqrt(np.mean(np.square(gsr_signal))),
            "gsr_range": np.ptp(gsr_signal)
        }
        
        #NaN Entfernen
        gsr_features = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) 
                       for k, v in gsr_features.items()}
        
        features.update(gsr_features)
        
    except (KeyError, ValueError) as e:
        print(f"GSR processing error: {str(e)}")
        features.update({k: 0.0 for k in [
            "gsr_mean", "gsr_std", "gsr_min", "gsr_max", 
            "gsr_variance", "gsr_skewness", "gsr_kurtosis",
            "gsr_abs_first_diff_mean", "gsr_abs_second_diff_mean",
            "gsr_mean_diff", "gsr_rms", "gsr_range"
        ]})

    # EMG Feature
    try:
        emg_signal = signal_dict["emg_trapezius"]
        features["emg_energy"] = np.sum(np.square(emg_signal)) if emg_signal.size else 0.0
    except KeyError:
        features["emg_energy"] = 0.0

    return features


def generate_data(data_dir="data/combined"):
    X = []
    y = []
    groups = [] 

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".csv"):
            continue

        subject_id = file_name[:6]
        groups.append(subject_id)

        if "-BL1-" in file_name:
            label = 0  # No pain
        elif "-PA4-" in file_name:
            label = 1  # Strong pain
        else:
            continue

        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)

        try:
            features = compute_features({
                "ecg": df["ecg"].values,
                "gsr": df["gsr"].values,
                "emg_trapezius": df["emg_trapezius"].values,
            })
        except KeyError as e:
            print(f"Missing column in {file_name}: {e}")
            continue

        X.append(features)
        y.append(label)

    X = pd.DataFrame(X)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups) 

    return X, y, groups

def main():
    print("\nGenerating data...\n")
    X, y, groups = generate_data() 
    
    # Split für RBFN und MLP
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("\nTraining and evaluating RBFN gamma=0.01, components=10.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=10, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.05, components=10.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.05, n_components=10, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.1, components=10.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=10, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.5, components=10.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.5, n_components=10, random_state=20)

    print("\nTraining and evaluating RBFN gamma=1, components=10.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=1, n_components=10, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.01, components=50.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=50, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.05, components=50.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.05, n_components=50, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.1, components=50.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=50, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.5, components=50.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.5, n_components=50, random_state=20)

    print("\nTraining and evaluating RBFN gamma=1, components=50.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=1, n_components=50, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.01, components=100.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=100, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.05, components=100.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.05, n_components=100, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.1, components=100.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=100, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.5, components=100.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.5, n_components=100, random_state=20)

    print("\nTraining and evaluating RBFN gamma=1, components=100.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=1, n_components=100, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.01, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=200, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.05, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.05, n_components=200, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.1, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=200, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.5, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.5, n_components=200, random_state=20)

    print("\nTraining and evaluating RBFN gamma=1, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=1, n_components=200, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.01, components=500.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.01, n_components=500, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.05, components=500.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.05, n_components=500, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.1, components=500.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=500, random_state=20)

    print("\nTraining and evaluating RBFN gamma=0.5, components=500.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.5, n_components=500, random_state=20)

    print("\nTraining and evaluating RBFN gamma=1, components=500.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=1, n_components=500, random_state=20)
    #train_mlp(X_train, X_test, y_train, y_test)
    
    #print("\nTraining Random Forest with Leave-One-Out...\n")
    #train_random_forest(X, y, groups)

    #print("\nTraining Support Vector Machine with Leave-One-Out...\n")
    #train_svm(X, y, groups)

if __name__ == "__main__":
    main()