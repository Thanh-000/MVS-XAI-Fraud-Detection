"""
Dataset loaders for the supported fraud benchmarks.

IEEE-CIS is loaded from the original Kaggle competition CSVs.
PaySim is canonicalized into the same core column names used by the pipeline so
the downstream feature engineering code can run with minimal branching.
"""
import os

import numpy as np
import pandas as pd


class DataLoader:
    """Load fraud datasets into a common tabular schema."""

    PAYSIM_CANDIDATES = (
        "paysim.csv",
        "PS_20174392719_1491204439457_log.csv",
        "paysim_log.csv",
    )

    @classmethod
    def load_dataset(cls, data_dir, dataset="ieee"):
        dataset = dataset.lower().strip()
        if dataset == "ieee":
            return cls._load_ieee(data_dir)
        if dataset == "paysim":
            return cls._load_paysim(data_dir)
        raise ValueError(f"Unsupported dataset '{dataset}'. Use 'ieee' or 'paysim'.")

    @classmethod
    def load_and_merge(cls, data_dir):
        """Backward-compatible IEEE-CIS entry point."""
        return cls.load_dataset(data_dir, dataset="ieee")

    @classmethod
    def _load_ieee(cls, data_dir):
        trans_path = os.path.join(data_dir, "train_transaction.csv")
        id_path = os.path.join(data_dir, "train_identity.csv")

        if not os.path.exists(trans_path) or not os.path.exists(id_path):
            raise FileNotFoundError(
                "IEEE-CIS requires 'train_transaction.csv' and 'train_identity.csv' "
                f"under {os.path.abspath(data_dir)}"
            )

        print("  Loading IEEE-CIS transaction data...")
        df_trans = pd.read_csv(trans_path)
        print(f"    Transactions: {df_trans.shape[0]:,} x {df_trans.shape[1]}")

        print("  Loading IEEE-CIS identity data...")
        df_id = pd.read_csv(id_path)
        print(f"    Identity: {df_id.shape[0]:,} x {df_id.shape[1]}")

        df = pd.merge(df_trans, df_id, on="TransactionID", how="left")
        print(f"    Merged: {df.shape[0]:,} x {df.shape[1]}")

        if "isFraud" in df.columns:
            df["isFraud"] = df["isFraud"].fillna(0).astype("int8")

        df = cls._reduce_mem_usage(df)
        return df.sort_values("TransactionDT").reset_index(drop=True)

    @classmethod
    def _load_paysim(cls, data_dir):
        paysim_path = cls._resolve_existing_path(data_dir, cls.PAYSIM_CANDIDATES)
        if paysim_path is None:
            raise FileNotFoundError(
                "PaySim file not found. Expected one of: "
                f"{', '.join(cls.PAYSIM_CANDIDATES)} under {os.path.abspath(data_dir)}"
            )

        print(f"  Loading PaySim data from '{os.path.basename(paysim_path)}'...")
        df = pd.read_csv(paysim_path)
        print(f"    Raw PaySim: {df.shape[0]:,} x {df.shape[1]}")

        required_cols = {
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        }
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(
                f"PaySim file is missing required columns: {', '.join(missing)}"
            )

        # Canonical columns used throughout the IEEE-oriented pipeline.
        df = df.copy()
        df["TransactionID"] = np.arange(1, len(df) + 1, dtype=np.int64)
        df["TransactionDT"] = df["step"].astype(np.int64) * 3600
        df["TransactionAmt"] = df["amount"].astype(np.float32)
        df["card1"] = pd.factorize(df["nameOrig"], sort=False)[0].astype(np.int32)
        df["addr1"] = pd.factorize(df["nameDest"], sort=False)[0].astype(np.int32)
        df["card4"] = df["type"].astype(str)
        df["D1"] = 0
        df["isFraud"] = df["isFraud"].fillna(0).astype("int8")

        # Lightweight balance-delta features give the tree models signal similar to
        # what IEEE identity/behavioral columns provide.
        df["origin_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["balance_gap"] = df["oldbalanceOrg"] - df["oldbalanceDest"]

        df = cls._reduce_mem_usage(df)
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        print(f"    Canonicalized PaySim: {df.shape[0]:,} x {df.shape[1]}")
        return df

    @staticmethod
    def _resolve_existing_path(data_dir, candidates):
        for filename in candidates:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _reduce_mem_usage(df):
        """Downcast numeric columns conservatively to reduce memory footprint."""
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == object:
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Float16 is too fragile for IEEE-CIS scale and can overflow during
                # casts or intermediate operations on Colab/NumPy. Float32 keeps the
                # memory savings while avoiding noisy warnings and precision loss.
                df[col] = df[col].astype(np.float32)

        end_mem = df.memory_usage().sum() / 1024**2
        reduction = (1 - end_mem / start_mem) * 100 if start_mem > 0 else 0
        print(f"    Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({reduction:.0f}% reduction)")
        return df
