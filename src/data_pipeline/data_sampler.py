"""
Data Balancing Engine — KMeansSMOTE and CTGAN synthesis.
Handles extreme class imbalance (3.5% fraud rate → target 30%).
"""
import pandas as pd
import numpy as np
import os

try:
    from imblearn.over_sampling import SMOTE, KMeansSMOTE, RandomOverSampler
except ImportError:
    SMOTE = None
    KMeansSMOTE = None
    RandomOverSampler = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError:
    MiniBatchKMeans = None

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

        n_rows = len(y_train)
        bincount = np.bincount(y_train)
        large_threshold = int(os.getenv("MVS_XAI_LARGE_SMOTE_ROWS", "1000000"))
        large_max_strategy = float(os.getenv("MVS_XAI_LARGE_SMOTE_MAX_STRATEGY", "0.10"))
        disable_large_ros = os.getenv("MVS_XAI_DISABLE_LARGE_ROS", "0") == "1"
        if n_rows >= large_threshold and RandomOverSampler is not None and not disable_large_ros:
            effective_strategy = min(float(strategy), large_max_strategy)
            print(
                f"  Large-data oversampling: RandomOverSampler "
                f"(strategy={effective_strategy}, rows={n_rows:,}); "
                "skipping KMeansSMOTE to avoid O(n*k*iter) K-Means cost. "
                "Set MVS_XAI_DISABLE_LARGE_ROS=1 to preserve KMeansSMOTE."
            )
            sampler = RandomOverSampler(
                sampling_strategy=effective_strategy,
                random_state=self.random_state,
            )
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            print(f"    Before: {bincount}")
            print(f"    After:  {np.bincount(y_res)}")
            return X_res, y_res

        print(f"  KMeansSMOTE (strategy={strategy})...")
        try:
            sampler_kwargs = {
                "sampling_strategy": strategy,
                "random_state": self.random_state,
                "k_neighbors": 5,
                "cluster_balance_threshold": 0.1,
            }
            use_minibatch = os.getenv("MVS_XAI_KMEANSSMOTE_MINIBATCH", "0") == "1"
            if use_minibatch and MiniBatchKMeans is not None:
                n_clusters = int(os.getenv("MVS_XAI_KMEANSSMOTE_CLUSTERS", "32"))
                batch_size = int(os.getenv("MVS_XAI_KMEANSSMOTE_BATCH_SIZE", "65536"))
                max_iter = int(os.getenv("MVS_XAI_KMEANSSMOTE_MAX_ITER", "100"))
                sampler_kwargs["kmeans_estimator"] = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    n_init="auto",
                    random_state=self.random_state,
                )
                print(
                    "    KMeansSMOTE accelerator: MiniBatchKMeans "
                    f"(clusters={n_clusters}, batch_size={batch_size:,}, max_iter={max_iter})"
                )
            sampler = KMeansSMOTE(
                **sampler_kwargs
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
