"""
View 1: Tabular Feature Extraction.
Matches notebook 06 — time-based features and NaN cleanup.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    def clean_high_nan_columns(df, threshold=0.7):
        """Drop columns with NaN ratio above threshold."""
        nan_ratio = df.isnull().mean()
        good_cols = nan_ratio[nan_ratio < threshold].index.tolist()
        n_dropped = len(df.columns) - len(good_cols)
        df = df[good_cols]
        print(f"  NaN cleanup: kept {len(good_cols)} columns (dropped {n_dropped} with NaN > {threshold*100:.0f}%)")
        return df

    @staticmethod
    def encode_categoricals(df):
        """Label-encode all object/string columns."""
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        return df
