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
        "paysim dataset.csv",
    )
    PAYSIM_REQUIRED_COLS = (
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
    )
    PAYSIM_OPTIONAL_COLS = ("isFlaggedFraud",)
    PAYSIM_DTYPES = {
        "step": np.int32,
        "amount": np.float32,
        "oldbalanceOrg": np.float32,
        "newbalanceOrig": np.float32,
        "oldbalanceDest": np.float32,
        "newbalanceDest": np.float32,
        "isFraud": np.int8,
        "isFlaggedFraud": np.int8,
    }

    @classmethod
    def load_dataset(cls, data_dir, dataset="ieee", **kwargs):
        dataset = dataset.lower().strip()
        if dataset == "ieee":
            return cls._load_ieee(data_dir)
        if dataset == "paysim":
            return cls._load_paysim(data_dir, **kwargs)
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
    def _load_paysim(
        cls,
        data_dir,
        chunk_size=None,
        max_rows=None,
        step_block_size=None,
    ):
        paysim_path = cls._resolve_paysim_path(data_dir)
        if paysim_path is None:
            raise FileNotFoundError(
                "PaySim file not found. Expected a CSV with required PaySim columns "
                f"under {os.path.abspath(data_dir)}"
            )

        print(f"  Loading PaySim data from '{os.path.basename(paysim_path)}'...")
        if chunk_size and chunk_size > 0:
            return cls._load_paysim_chunked(
                paysim_path,
                chunk_size=int(chunk_size),
                max_rows=max_rows,
                step_block_size=step_block_size,
            )

        df = pd.read_csv(paysim_path)
        print(f"    Raw PaySim: {df.shape[0]:,} x {df.shape[1]}")

        missing = sorted(set(cls.PAYSIM_REQUIRED_COLS) - set(df.columns))
        if missing:
            raise ValueError(
                f"PaySim file is missing required columns: {', '.join(missing)}"
            )

        df = cls._canonicalize_paysim(df, step_block_size=step_block_size)
        df = cls._reduce_mem_usage(df)
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        print(f"    Canonicalized PaySim: {df.shape[0]:,} x {df.shape[1]}")
        return df

    @classmethod
    def _load_paysim_chunked(cls, paysim_path, chunk_size, max_rows=None, step_block_size=None):
        header = pd.read_csv(paysim_path, nrows=0)
        missing = sorted(set(cls.PAYSIM_REQUIRED_COLS) - set(header.columns))
        if missing:
            raise ValueError(
                f"PaySim file is missing required columns: {', '.join(missing)}"
            )

        present_cols = list(cls.PAYSIM_REQUIRED_COLS) + [
            col for col in cls.PAYSIM_OPTIONAL_COLS if col in header.columns
        ]
        dtype_map = {col: dtype for col, dtype in cls.PAYSIM_DTYPES.items() if col in present_cols}

        print(
            f"    PaySim large-dataset mode: chunk_size={chunk_size:,}"
            + (f", max_rows={int(max_rows):,}" if max_rows else "")
            + (f", step_block_size={int(step_block_size)}" if step_block_size else "")
        )

        origin_map = {}
        dest_map = {}
        next_origin_id = 0
        next_dest_id = 0
        row_offset = 0
        chunks = []

        reader = pd.read_csv(
            paysim_path,
            usecols=present_cols,
            dtype=dtype_map,
            chunksize=chunk_size,
        )
        for chunk_id, chunk in enumerate(reader, start=1):
            if max_rows is not None and row_offset >= int(max_rows):
                break

            if max_rows is not None and row_offset + len(chunk) > int(max_rows):
                chunk = chunk.iloc[: int(max_rows) - row_offset].copy()

            chunk, next_origin_id = cls._encode_with_mapping(chunk, "nameOrig", "card1", origin_map, next_origin_id)
            chunk, next_dest_id = cls._encode_with_mapping(chunk, "nameDest", "addr1", dest_map, next_dest_id)
            chunk = cls._canonicalize_paysim(
                chunk,
                start_id=row_offset + 1,
                preserve_existing_ids=True,
                step_block_size=step_block_size,
            )
            chunks.append(chunk)
            row_offset += len(chunk)
            print(
                f"    Chunk {chunk_id}: {len(chunk):,} rows "
                f"(cumulative {row_offset:,})"
            )

        if not chunks:
            raise ValueError("PaySim chunked loader produced no rows.")

        df = pd.concat(chunks, ignore_index=True)
        df = cls._reduce_mem_usage(df)
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        print(f"    Canonicalized PaySim: {df.shape[0]:,} x {df.shape[1]}")
        return df

    @classmethod
    def _canonicalize_paysim(
        cls,
        df,
        start_id=1,
        preserve_existing_ids=False,
        step_block_size=None,
    ):
        """Map PaySim into the common fraud schema used by the pipeline."""
        df = df.copy()
        if not preserve_existing_ids:
            if "card1" not in df.columns:
                df["card1"] = pd.factorize(df["nameOrig"], sort=False)[0].astype(np.int32)
            if "addr1" not in df.columns:
                df["addr1"] = pd.factorize(df["nameDest"], sort=False)[0].astype(np.int32)

        df["TransactionID"] = np.arange(start_id, start_id + len(df), dtype=np.int64)
        df["TransactionDT"] = df["step"].astype(np.int64) * 3600
        df["TransactionAmt"] = df["amount"].astype(np.float32)
        df["card4"] = df["type"].astype(str)
        df["D1"] = 0
        df["isFraud"] = df["isFraud"].fillna(0).astype("int8")
        df["origin_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["dest_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["balance_gap"] = df["oldbalanceOrg"] - df["oldbalanceDest"]

        if step_block_size and step_block_size > 0:
            df["StepBlock"] = (df["step"].astype(np.int64) // int(step_block_size)).astype(np.int32)

        return df

    @staticmethod
    def _encode_with_mapping(df, source_col, target_col, mapping, next_id):
        values = df[source_col].astype(str)
        uniques = pd.Index(values.unique())
        new_values = [value for value in uniques if value not in mapping]
        if new_values:
            mapping.update({value: next_id + idx for idx, value in enumerate(new_values)})
            next_id += len(new_values)
        df[target_col] = values.map(mapping).astype(np.int32)
        return df, next_id

    @staticmethod
    def _resolve_existing_path(data_dir, candidates):
        for filename in candidates:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def _resolve_paysim_path(cls, data_dir):
        candidate = cls._resolve_existing_path(data_dir, cls.PAYSIM_CANDIDATES)
        if candidate is not None:
            return candidate

        for root, _, _ in os.walk(data_dir):
            candidate = cls._resolve_existing_path(root, cls.PAYSIM_CANDIDATES)
            if candidate is not None:
                return candidate

        required = set(cls.PAYSIM_REQUIRED_COLS)
        for root, _, files in os.walk(data_dir):
            for filename in sorted(files):
                if not filename.lower().endswith(".csv"):
                    continue
                path = os.path.join(root, filename)
                try:
                    columns = set(pd.read_csv(path, nrows=0).columns)
                except Exception:
                    continue
                if required.issubset(columns):
                    return path
        return None

    @staticmethod
    def _reduce_mem_usage(df):
        """Downcast numeric columns conservatively to reduce memory footprint."""
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtype
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(df[col]):
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
