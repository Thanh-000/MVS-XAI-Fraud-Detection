"""
Data Balancing Engine — KMeansSMOTE and CTGAN synthesis.
Handles extreme class imbalance (3.5% fraud rate → target 30%).
"""
import pandas as pd
import numpy as np

try:
    from imblearn.over_sampling import SMOTE, KMeansSMOTE
except ImportError:
    SMOTE = None
    KMeansSMOTE = None

try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
except ImportError:
    CTGANSynthesizer = None


class DataBalanceEngine:
    """Data augmentation for imbalanced binary classification.

    Strategy (as per proposal): KMeansSMOTE for fast augmentation,
    CTGAN for generating realistic synthetic fraud samples.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def apply_kmeans_smote(self, X_train, y_train, strategy=0.3):
        """Apply KMeansSMOTE with fallback to standard SMOTE.

        Clusters minority samples before interpolation for more
        realistic synthetic examples.

        Args:
            X_train: Training features.
            y_train: Training labels.
            strategy: Target minority/majority ratio (default: 0.3 = 30%).

        Returns:
            Resampled (X, y) tuple.
        """
        if KMeansSMOTE is None or SMOTE is None:
            raise ImportError("KMeansSMOTE/SMOTE requires imbalanced-learn: pip install imbalanced-learn")

        print(f"  KMeansSMOTE (strategy={strategy})...")
        try:
            sampler = KMeansSMOTE(
                sampling_strategy=strategy,
                random_state=self.random_state,
                k_neighbors=5,
                cluster_balance_threshold=0.1
            )
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"    KMeansSMOTE failed ({e}), falling back to SMOTE")
            sampler = SMOTE(
                sampling_strategy=strategy,
                random_state=self.random_state
            )
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        print(f"    Before: {np.bincount(y_train)}")
        print(f"    After:  {np.bincount(y_res)}")
        return X_res, y_res

    def apply_ctgan_synthesis(self, df_train, target_col='isFraud',
                              num_synthetic_samples=10000, epochs=30, use_gpu=True):
        """Generate synthetic fraud samples using CTGAN.

        Args:
            df_train: Training DataFrame.
            target_col: Target column name.
            num_synthetic_samples: Number of synthetic samples to generate.
            epochs: CTGAN training epochs.

        Returns:
            Augmented DataFrame.
        """
        if CTGANSynthesizer is None:
            raise ImportError("CTGAN requires SDV: pip install sdv")

        print(f"  CTGAN Synthesis (epochs={epochs}, samples={num_synthetic_samples:,})...")
        fraud_data = df_train[df_train[target_col] == 1]

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=fraud_data)

        for col_name in fraud_data.columns:
            if col_name != target_col:
                metadata.update_column(column_name=col_name, sdtype='numerical')

        synthesizer = CTGANSynthesizer(
            metadata, epochs=epochs, verbose=True, cuda=use_gpu
        )
        synthesizer.fit(fraud_data)
        synthetic_data = synthesizer.sample(num_rows=num_synthetic_samples)

        if target_col not in synthetic_data.columns:
            synthetic_data[target_col] = 1

        augmented_df = pd.concat([df_train, synthetic_data], axis=0).sample(
            frac=1.0, random_state=self.random_state
        )
        print(f"    Added {num_synthetic_samples:,} synthetic fraud samples")
        return augmented_df
