import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import src.visualize as vis
from sklearn.metrics import f1_score

def f1_threshold_scorer(threshold: float = 0.5):
    """Gibt einen Callable zurück, der für permutation_importance passt."""
    def _score(estimator, X, y_true):
        # kontinuierliche Scores → Klassenlabels
        y_scores = estimator.predict(X)
        y_pred = (y_scores >= threshold).astype(int)
        return f1_score(y_true, y_pred)
    return _score

def permutation_f1_importance(
    model,
    X_val: pd.DataFrame,
    y_val,
    *,
    n_repeats: int = 30,
    threshold: float = 0.5,
    random_state: int | None = 42,
):
    scorer = f1_threshold_scorer(threshold)          # ← neuer Scorer
    
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scorer,          # ← hier einsetzen
        n_repeats=n_repeats,
        random_state=random_state,
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
    random_state,
    threshold=0.5
):
    
    
    model = Pipeline([
    ("scaler", StandardScaler()),
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

    perm_df = permutation_f1_importance(
        model, X_test, y_test, n_repeats=30, random_state=random_state
    )
    print("\nPermutation importance (ΔF1):\n", perm_df)

    # (optional) save results
    perm_df.to_csv("rbfn_perm_importance.csv", index=False)

    #vis.plot_confusion_matrix(y_test, y_pred_class)
    #vis.plot_roc_curve(y_test, y_scores)
    #vis.plot_prediction_distribution(y_test, y_scores)
    
    return model, y_scores, y_pred_class