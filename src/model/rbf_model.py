import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import src.visualize as vis
import os     
from sklearn.metrics import f1_score, precision_score, recall_score

def f1_threshold_scorer(threshold: float = 0.5):
    def _score(estimator, X, y_true):
        y_scores = estimator.predict(X)
        y_pred = (y_scores >= threshold).astype(int)
        return f1_score(y_true, y_pred)
    return _score

def permutation_f1_importance(model, X_val: pd.DataFrame, y_val, *, n_repeats: int = 30, threshold: float = 0.5):
    scorer = f1_threshold_scorer(threshold)         
    
    result = permutation_importance(model,X_val, y_val, scoring=scorer,n_repeats=n_repeats,
        n_jobs=-1,
    )

    return (
        pd.DataFrame(
            {
                "feature": X_val.columns,
                "delta_f1": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("delta_f1", ascending=False)
        .reset_index(drop=True)
    )

def train_and_test_sklearn_rbf(
    X_train,
    y_train,
    X_test,
    y_test,
    gamma,
    n_components,
    threshold=0.5
):
    
    
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf_feature", RBFSampler(gamma=gamma, n_components=n_components)),
    ("linear", Ridge())
])
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_scores = model.predict(X_test)
    y_pred_class = np.where(y_scores >= threshold, 1, 0)
    
    acc      = accuracy_score(y_test, y_pred_class)
    prec     = precision_score(y_test, y_pred_class, zero_division=0)
    rec      = recall_score(y_test, y_pred_class,    zero_division=0)
    f1       = f1_score(y_test, y_pred_class)

    # Metriken ausgeben
    print("Accuracy RBF:", accuracy_score(y_test, y_pred_class))
    print("Classification report RBF:\n", classification_report(y_test, y_pred_class))

    results_df = pd.DataFrame([{
        "precision": prec,
        "recall"   : rec,
        "accuracy" : acc,
        "f1_score" : f1
    }])

    csv_path = "rbf_run_metrics.csv"        # beliebiger Dateiname

    # Wenn Datei schon existiert → anhängen ohne Header,
    # sonst neu anlegen mit Header
    if os.path.exists(csv_path):
        results_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(csv_path, mode="w", header=True,  index=False)


    #perm_df = permutation_f1_importance(
    #    model, X_test, y_test, n_repeats=30
    #)
    #print("\nPermutation importance (ΔF1):\n", perm_df)

    # (optional) save results
    #perm_df.to_csv("rbfn_perm_importance.csv", index=False)

    #vis.plot_confusion_matrix(y_test, y_pred_class)
    #vis.plot_roc_curve(y_test, y_scores)
    #vis.plot_prediction_distribution(y_test, y_scores)
    
    return model, y_scores, y_pred_class