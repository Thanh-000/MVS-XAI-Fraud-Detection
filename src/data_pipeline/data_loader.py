"""
Data Loader — Merge Transaction and Identity tables.
Includes memory reduction optimization from Kaggle kernels.
"""
import pandas as pd
import numpy as np
import os


class DataLoader:
    """Load and merge IEEE-CIS Transaction + Identity datasets."""

    @staticmethod
    def load_and_merge(data_dir):
        """Load and merge Transaction + Identity CSVs with memory optimization.

        Args:
            data_dir: Directory containing train_transaction.csv and train_identity.csv.

        Returns:
            Merged DataFrame.
        """
        trans_path = os.path.join(data_dir, 'train_transaction.csv')
        id_path = os.path.join(data_dir, 'train_identity.csv')

        print("  Loading Transaction data...")
        df_trans = pd.read_csv(trans_path)
        print(f"    Transactions: {df_trans.shape[0]:,} × {df_trans.shape[1]}")

        print("  Loading Identity data...")
        df_id = pd.read_csv(id_path)
        print(f"    Identity: {df_id.shape[0]:,} × {df_id.shape[1]}")

        # Left join on TransactionID (not all transactions have identity info)
        df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
        print(f"    Merged: {df.shape[0]:,} × {df.shape[1]}")

        # Ensure target column is clean
        if 'isFraud' in df.columns:
            df['isFraud'] = df['isFraud'].fillna(0).astype('int8')

        # Memory optimization
        df = DataLoader._reduce_mem_usage(df)
        return df

    @staticmethod
    def _reduce_mem_usage(df):
        """Downcast numeric columns to reduce memory footprint.

        Standard Kaggle technique for large datasets.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    else:
                        df[col] = df[col].astype(np.float32)
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"    Memory: {start_mem:.1f} MB → {end_mem:.1f} MB "
              f"({(1-end_mem/start_mem)*100:.0f}% reduction)")
        return df
