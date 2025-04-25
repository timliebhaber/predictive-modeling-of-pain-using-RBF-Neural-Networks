import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import neurokit2 as nk
from sklearn.inspection import permutation_importance


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.mlp_model import train_mlp
from src.model.random_forest_model import train_random_forest
from src.model.rbf_model import train_and_test_sklearn_rbf
from src.model.svm_model import train_svm
from src.rbf_feature_stats import feature_stats

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

    bl1_count = 0
    pa4_count = 0

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".csv"):
            continue

        subject_id = file_name[:6]
        groups.append(subject_id)

        if "-BL1-" in file_name:
            label = 0  # No pain
            bl1_count += 1
        elif "-PA4-" in file_name:
            label = 1  # Strong pain
            pa4_count += 1
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

    print(f"Amount BL1-Labels (no pain): {bl1_count}")
    print(f"Amount PA4-Labels (severe pain): {pa4_count}")

    X = pd.DataFrame(X)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups) 

    return X, y, groups

def run_significance_tests(
    X: pd.DataFrame,
    y: np.ndarray,
    alpha: float = 0.05,
    fdr_correction: bool = True,
) -> pd.DataFrame:
    """
    Vergleicht BL1- (y==0) und PA4-Gruppe (y==1) für jedes Feature:
    1. Shapiro-Wilk: Normalität pro Gruppe
    2. t-Test (unequal var) bei Normalität, sonst Mann-Whitney-U
    3. Optionale FDR-Korrektur (Benjamini–Hochberg)
    4. Effekt­stärke:  Cohen's d   (param.) bzw. Rank-Biserial (non-param.)
    Gibt einen DataFrame mit allen Kennzahlen zurück.
    """
    results = []

    for feat in X.columns:
        g0 = X.loc[y == 0, feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        g1 = X.loc[y == 1, feat].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        # Falls zu wenige Beobachtungen vorhanden sind, überspringen
        if min(len(g0), len(g1)) < 3:
            results.append(
                dict(
                    feature=feat,
                    test="NA",
                    stat=np.nan,
                    p_value=np.nan,
                    effect_size=np.nan,
                    mean_bl1=g0.mean(),
                    sd_bl1=g0.std(ddof=1),
                    median_bl1=g0.median(),
                    iqr_bl1=g0.quantile(0.75) - g0.quantile(0.25),
                    mean_pa4=g1.mean(),
                    sd_pa4=g1.std(ddof=1),
                    median_pa4=g1.median(),
                    iqr_pa4=g1.quantile(0.75) - g1.quantile(0.25),
                )
            )
            continue

        # 1) Shapiro-Wilk
        p_norm_g0 = stats.shapiro(g0).pvalue if len(g0) < 5000 else 1.0  # SciPy-Limitation
        p_norm_g1 = stats.shapiro(g1).pvalue if len(g1) < 5000 else 1.0
        normal = (p_norm_g0 > alpha) and (p_norm_g1 > alpha)

        # 2) Parametrisch vs. nicht-parametrisch
        if normal:
            stat, p_val = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
            test_name = "t-test"

            # Cohen's d (pooled SD)
            pooled_sd = np.sqrt(
                ((len(g0) - 1) * g0.var(ddof=1) + (len(g1) - 1) * g1.var(ddof=1))
                / (len(g0) + len(g1) - 2)
            )
            eff = (g0.mean() - g1.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        else:
            stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            test_name = "Mann-Whitney-U"

            # Rank-Biserial-Korrelations­koeffizient als Effekt
            n1, n2 = len(g0), len(g1)
            eff = 1 - (2 * stat) / (n1 * n2)  # range −1…+1

        results.append(
            dict(
                feature=feat,
                test=test_name,
                stat=stat,
                p_value=p_val,
                effect_size=eff,
                mean_bl1=g0.mean(),
                sd_bl1=g0.std(ddof=1),
                median_bl1=g0.median(),
                iqr_bl1=g0.quantile(0.75) - g0.quantile(0.25),
                mean_pa4=g1.mean(),
                sd_pa4=g1.std(ddof=1),
                median_pa4=g1.median(),
                iqr_pa4=g1.quantile(0.75) - g1.quantile(0.25),
            )
        )

    res_df = pd.DataFrame(results)

    # 3) Mehrfach­korrektur (Benjamini-Hochberg)
    if fdr_correction:
        mask = res_df["p_value"].notna()
        res_df.loc[mask, "p_adj"] = multipletests(
            res_df.loc[mask, "p_value"], alpha=alpha, method="fdr_bh"
        )[1]
        res_df["significant"] = res_df["p_adj"] < alpha
    else:
        res_df["p_adj"] = np.nan
        res_df["significant"] = res_df["p_value"] < alpha

    return res_df



def main():
    print("\nGenerating data...\n")
    X, y, groups = generate_data() 

    # -------------------------------------------------------------------------
    # 1) EDA: Lage- & Streuungs­parameter (gab es schon in feature_stats)
    # -------------------------------------------------------------------------
    stats_df = feature_stats(X)
    print("\nRaw-feature means & stds:\n", stats_df)
    stats_df.to_csv("raw_feature_stats.csv", index=True)

    print("\nRunning significance tests (Shapiro → t-Test / Mann-Whitney)…\n")
    sig_df = run_significance_tests(X, y, alpha=0.05, fdr_correction=True)
    print(sig_df[["feature", "test", "p_value", "p_adj", "significant"]])

    # CSV für spätere LaTeX-Tabelle
    sig_df.to_csv("feature_significance.csv", index=False)
    print("\nSignificance results written to feature_significance.csv")

    stats = feature_stats(X)
    print("\nRaw-feature means & stds:\n", stats)
    stats.to_csv("raw_feature_stats.csv", index=True)
    
    # Split für RBFN und MLP
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("\nTraining and evaluating RBFN gamma=0.1, components=200.\n")
    train_and_test_sklearn_rbf(X_train, y_train, X_test, y_test, gamma=0.1, n_components=200, random_state=20)

    #train_mlp(X_train, X_test, y_train, y_test)
    
    #print("\nTraining Random Forest with Leave-One-Out...\n")
    #train_random_forest(X, y, groups)

    #print("\nTraining Support Vector Machine with Leave-One-Out...\n")
    #train_svm(X, y, groups)

if __name__ == "__main__":
    main()