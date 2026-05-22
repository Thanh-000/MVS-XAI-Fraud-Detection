"""
View 1: Tabular Feature Extraction.
Matches notebook 06 — time-based features and NaN cleanup.
"""
import pandas as pd
import numpy as np


class TabularFeatureExtractor:
    """Extract time-based and ratio features from raw transaction data."""

    @staticmethod
    def extract_time_features(df):
        """Derive hour, day_of_week, is_night, is_weekend from TransactionDT."""
        transaction_dt = df['TransactionDT'].astype(np.int64)
        df['hour'] = (transaction_dt % 86400) // 3600
        df['day_of_week'] = (transaction_dt // 86400) % 7
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        print("  View 1 (Tabular): hour, day_of_week, is_night, is_weekend")
        return df

    @staticmethod
    def clean_high_nan_columns(df, threshold=0.7, fit_idx=None, preserve_cols=None):
        """Drop columns with NaN ratio above threshold using train-only fit rows."""
        if fit_idx is None:
            fit_frame = df
        else:
            fit_frame = df.iloc[fit_idx]

        preserve_cols = set(preserve_cols or [])
        nan_ratio = fit_frame.isnull().mean()
        good_cols = nan_ratio[nan_ratio < threshold].index.tolist()
        good_cols = list(dict.fromkeys(good_cols + [c for c in preserve_cols if c in df.columns]))
        n_dropped = len(df.columns) - len(good_cols)
        df = df[good_cols]
        scope = "train slice" if fit_idx is not None else "full data"
        print(
            f"  NaN cleanup: kept {len(good_cols)} columns "
            f"(dropped {n_dropped} with NaN > {threshold*100:.0f}%, fit={scope})"
        )
        return df

    @staticmethod
    def encode_categoricals(df, fit_idx=None):
        """Label-encode object/string columns with mappings learned on fit rows only."""
        if fit_idx is None:
            fit_frame = df
        else:
            fit_frame = df.iloc[fit_idx]

        for c in df.select_dtypes(include=['object', 'string']).columns:
            train_values = fit_frame[c].fillna("__MISSING__").astype(str)
            classes = pd.Index(train_values.unique())
            values = df[c].fillna("__MISSING__").astype(str).to_numpy()
            df[c] = classes.get_indexer(values).astype(np.int32)
        return df
