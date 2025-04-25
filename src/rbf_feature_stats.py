import pandas as pd

def feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    
    if X.shape[1] != 14:
            raise ValueError(
                f"Expected 14 features, but X has {X.shape[1]}. "
                "Make sure you passed the raw handcrafted-feature table."
            )

    stats_df = X.agg(['mean', 'std']).T   # rows → features, columns → stats
    stats_df.columns = ['mean', 'std']
    return stats_df