import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred_class):
    """
    Erstellt eine Heatmap der Konfusionsmatrix, die die Verteilung der wahren und vorhergesagten Klassen zeigt.
    """
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Wahre Klasse")
    plt.title("Konfusionsmatrix")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_scores):
    """
    Plottet die ROC-Kurve (Receiver Operating Characteristic) und berechnet die AUC (Area Under the Curve),
    basierend auf den kontinuierlichen Vorhersage-Scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Zufall")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_prediction_distribution(y_true, y_scores):
    """
    Plottet die Verteilung der vom Modell gelieferten Vorhersage-Scores getrennt nach den wahren Klassen.
    So erhält man einen Eindruck, wie gut die Score-Verteilungen der Klassen getrennt sind.
    """
    plt.figure(figsize=(8, 6))
    # Aufteilung der Scores nach den wahren Klassen
    scores_class0 = y_scores[y_true == 0]
    scores_class1 = y_scores[y_true == 1]
    
    plt.hist(scores_class0, bins=50, alpha=0.6, label="Wahre Klasse 0")
    plt.hist(scores_class1, bins=50, alpha=0.6, label="Wahre Klasse 1")
    plt.xlabel("Vorhersage-Score")
    plt.ylabel("Häufigkeit")
    plt.title("Verteilung der Vorhersage-Scores")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
