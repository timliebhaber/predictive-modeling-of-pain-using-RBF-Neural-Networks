import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import src.visualize as vis

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

    vis.plot_confusion_matrix(y_test, y_pred_class)
    #vis.plot_roc_curve(y_test, y_scores)
    #vis.plot_prediction_distribution(y_test, y_scores)
    
    return model, y_scores, y_pred_class