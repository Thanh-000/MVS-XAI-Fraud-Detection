"""
Kaggle Winner Feature Engineering — UID Construction & Aggregations.
Based on Chris Deotte / FraudSquad (1st place, IEEE-CIS 2019, 6,381 teams).
Source: NVIDIA Developer Blog + Kaggle Discussion.

Key Insight:
  D1 = number of days since client started using the card.
  D1n = TransactionDT - D1 → near-constant per client (client "birthday").
  UID = card1 + addr1 + D1n → stable pseudo-client-ID across time.

This technique achieved AUC 0.947 on IEEE-CIS leaderboard.

Features created from UID:
  - Transaction aggregations: TransactionAmt (mean, std, z-score)
  - Categorical aggregations: M9_uid_mean, C13_uid_mean
  - D-column aggregations: D1/D15 per-UID diff, pct_change
  - V-column PCA: group by NaN pattern → PCA per group
  - Cross-feature interactions: Amt×hour, Amt×weekend, Amt×night

Post-processing:
  - UID-average prediction smoothing (replace individual → UID mean)
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class UIDFeatureEngineer:
    """Kaggle-grade feature engineering for IEEE-CIS fraud detection.

    Core idea: card1 + addr1 + D1_normalized creates a stable
    pseudo-client-ID that persists across transactions, even when
    the hacker changes IP/device/email.
    """

    @staticmethod
    def build_uid(df):
        """Construct client-level UID from card1 + addr1 + D1_normalized.

        D1_normalized (D1n) = TransactionDT - D1:
          - D1 = days since card was first used
          - TransactionDT = seconds since reference time
          - D1n = reference "birthday" of the card → near-constant per client
          - This creates a stable fingerprint even across different sessions

        Why this works:
          - card1 alone has ~13k unique values (too granular for some, too coarse for others)
          - Adding addr1 splits shared cards across different addresses
          - Adding D1n further splits by card issuance time → unique per person
        """
        # Step 1: Compute D1_normalized (card "birthday")
        if 'D1' in df.columns and 'TransactionDT' in df.columns:
            # D1 is in days, TransactionDT in seconds → convert to same unit
            df['D1n'] = (df['TransactionDT'] / 86400).round() - df['D1']
            # Round to integer to group slight variations
            df['D1n'] = df['D1n'].fillna(-999).astype(int)
        else:
            df['D1n'] = -999

        # Step 2: Build composite UID
        df['UID'] = (
            df['card1'].astype(str) + '_' +
            df['addr1'].fillna(-1).astype(int).astype(str) + '_' +
            df['D1n'].astype(str)
        )

        n_unique = df['UID'].nunique()
        n_total = len(df)
        print(f"  UID Construction: card1 + addr1 + D1n -> {n_unique:,} unique clients "
              f"({n_unique/n_total*100:.1f}% granularity)")
        return df

    @staticmethod
    def uid_aggregations(df):
        """Compute per-UID aggregated features using expanding window (no leakage).

        Creates features that capture client-level behavioral patterns:
        - TransactionAmt: mean, std, count, z-score per client
        - C-columns (C1, C13, C14): counting features per client
        - M-columns (M4, M5, M6, M9): match features per client
        - D-columns (D1, D15): temporal features per client

        Uses .expanding() (backward-only) to prevent future data leakage.
        """
        df = df.sort_values('TransactionDT').reset_index(drop=True)
        print("  UID Aggregations (expanding, no future leakage):")

        # --- TransactionAmt aggregations (most important) ---
        grp_amt = df.groupby('UID')['TransactionAmt']
        df['UID_amt_mean'] = grp_amt.transform(lambda x: x.expanding().mean())
        df['UID_amt_std'] = grp_amt.transform(lambda x: x.expanding().std()).fillna(0)
        df['UID_txn_count'] = grp_amt.transform(lambda x: x.expanding().count())
        df['UID_amt_zscore'] = (
            (df['TransactionAmt'] - df['UID_amt_mean']) / (df['UID_amt_std'] + 1e-6)
        )
        print(f"    TransactionAmt: mean, std, count, z-score")

        # --- C-column aggregations (counting features) ---
        for c_col in ['C1', 'C13', 'C14']:
            if c_col not in df.columns:
                continue
            g = df.groupby('UID')[c_col]
            df[f'{c_col}_uid_mean'] = g.transform(lambda x: x.expanding().mean())
            df[f'{c_col}_uid_std'] = g.transform(lambda x: x.expanding().std()).fillna(0)
        print(f"    C-columns: C1, C13, C14 per-UID mean/std")

        # --- M-column aggregations (match/mismatch features) ---
        for m_col in ['M4', 'M5', 'M6', 'M9']:
            if m_col not in df.columns:
                continue
            # M-columns may be categorical (T/F) → encode first
            if df[m_col].dtype == object:
                df[m_col] = df[m_col].map({'T': 1, 'F': 0}).fillna(-1)
            g = df.groupby('UID')[m_col]
            df[f'{m_col}_uid_mean'] = g.transform(lambda x: x.expanding().mean())
        print(f"    M-columns: M4, M5, M6, M9 per-UID mean "
              f"(M9_uid_mean is top Kaggle feature)")

        # --- D-column aggregations (temporal features) ---
        for d_col in ['D1', 'D15']:
            if d_col not in df.columns:
                continue
            g = df.groupby('UID')[d_col]
            df[f'{d_col}_uid_mean'] = g.transform(lambda x: x.expanding().mean())
            df[f'{d_col}_uid_std'] = g.transform(lambda x: x.expanding().std()).fillna(0)
            df[f'{d_col}_uid_diff'] = g.diff().fillna(0)
            df[f'{d_col}_uid_pct'] = g.pct_change().replace(
                [np.inf, -np.inf], 0
            ).fillna(0)
        print(f"    D-columns: D1, D15 per-UID mean/std/diff/pct_change")

        return df

    @staticmethod
    def v_column_pca(df, n_components=3):
        """Group anonymous V-columns by NaN pattern, then PCA per group.

        V1–V339 are anonymous features with correlated groups sharing
        similar NaN patterns. PCA reduces dimensionality while preserving signal.
        """
        v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
        if len(v_cols) < 10:
            print(f"  V-PCA: Skipped (only {len(v_cols)} V-columns)")
            return df

        # Group V-columns by NaN pattern
        nan_patterns = df[v_cols].isnull().astype(int)
        pattern_groups = {}
        for col in v_cols:
            pattern_key = tuple(nan_patterns[col].values[:100])  # Sample for efficiency
            if pattern_key not in pattern_groups:
                pattern_groups[pattern_key] = []
            pattern_groups[pattern_key].append(col)

        n_pca_features = 0
        for gi, (_, group_cols) in enumerate(pattern_groups.items()):
            if len(group_cols) < n_components:
                continue
            sub = df[group_cols].fillna(0).values
            n_comp = min(n_components, len(group_cols), sub.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            pca_result = pca.fit_transform(sub)
            for pc_i in range(n_comp):
                df[f'V_PCA_g{gi}_pc{pc_i}'] = pca_result[:, pc_i]
            n_pca_features += n_comp

        # Drop original V-columns to reduce noise
        df.drop(columns=v_cols, inplace=True, errors='ignore')
        print(f"  V-PCA: {len(v_cols)} V-columns -> {n_pca_features} PCA components "
              f"({len(pattern_groups)} groups)")
        return df

    @staticmethod
    def cross_feature_interactions(df):
        """Create TransactionAmt × time-of-day interaction features.

        Fraud patterns differ significantly by time of day:
        - Night transactions with high amounts → suspicious
        - Weekend + large amount → unusual for corporate cards
        """
        if 'hour' in df.columns:
            df['Amt_x_Hour'] = df['TransactionAmt'] * df['hour']
            df['Amt_x_isWeekend'] = df['TransactionAmt'] * df.get('is_weekend', 0)
            df['Amt_x_isNight'] = df['TransactionAmt'] * df.get('is_night', 0)
            print("  Cross-features: Amt*Hour, Amt*Weekend, Amt*Night")
        return df

    @staticmethod
    def new_client_flag(df):
        """Flag clients with fewer than 2 transactions (cold-start detection)."""
        if 'UID_txn_count' in df.columns:
            df['is_new_client'] = (df['UID_txn_count'] < 2).astype(int)
            n_new = df['is_new_client'].sum()
            print(f"  New client flag: {n_new:,} cold-start transactions "
                  f"({n_new/len(df)*100:.1f}%)")
        return df

    @classmethod
    def apply_all(cls, df, dataset_name='ieee'):
        """Apply all Kaggle Winner feature engineering steps sequentially.

        Pipeline: build_uid → uid_aggregations → d_diffs (in agg) →
                  v_pca → cross_features → new_client_flag
        """
        print("=" * 60)
        if dataset_name.lower() == 'ieee':
            print("  Kaggle Winner Feature Engineering (v4.3.4)")
            print("  Source: Chris Deotte / FraudSquad / NVIDIA Developer")
        else:
            print(f"  Canonical Feature Engineering for {dataset_name.upper()} (v4.3.4)")
            print("  Reuses the common UID/aggregation stack on dataset-adapted columns")
        print("=" * 60)

        df = cls.build_uid(df)
        df = cls.uid_aggregations(df)
        df = cls.v_column_pca(df)
        df = cls.cross_feature_interactions(df)
        df = cls.new_client_flag(df)

        print(f"\n  Total columns after FE: {len(df.columns)}")
        return df


class UIDPostProcessor:
    """Post-processing: Replace individual predictions with UID-average.

    Key Kaggle insight: If multiple transactions belong to the same UID,
    they should all have similar fraud probability. Averaging predictions
    per UID reduces noise and improves AUC by ~0.002-0.005.

    This is the FINAL step after meta-learner prediction, before thresholding.
    """

    @staticmethod
    def uid_average_predictions(df, pred_col='fraud_score', uid_col='UID',
                                blend_ratio=0.7):
        """Replace individual predictions with blended UID-average.

        final_score = blend_ratio × individual_score +
                      (1 - blend_ratio) × uid_mean_score

        Args:
            df: DataFrame with predictions and UID column.
            pred_col: Name of the prediction score column.
            uid_col: Name of the UID column.
            blend_ratio: Weight for individual score (default: 0.7).
                         0.7 means 70% individual + 30% UID average.

        Returns:
            DataFrame with smoothed prediction scores.
        """
        uid_mean = df.groupby(uid_col)[pred_col].transform('mean')
        uid_count = df.groupby(uid_col)[pred_col].transform('count')

        # Only apply smoothing for UIDs with 2+ transactions
        # Single-transaction UIDs keep their original score
        mask_multi = uid_count >= 2

        original_scores = df[pred_col].copy()
        df[pred_col] = np.where(
            mask_multi,
            blend_ratio * df[pred_col] + (1 - blend_ratio) * uid_mean,
            df[pred_col]
        )

        n_smoothed = mask_multi.sum()
        mean_delta = (df[pred_col] - original_scores).abs().mean()
        print(f"\n  UID Post-Processing:")
        print(f"    Blend ratio: {blend_ratio:.0%} individual + {1-blend_ratio:.0%} UID mean")
        print(f"    Smoothed: {n_smoothed:,} / {len(df):,} transactions "
              f"({n_smoothed/len(df)*100:.1f}%)")
        print(f"    Mean score change: {mean_delta:.4f}")

        return df

    @staticmethod
    def uid_full_average(predictions, uid_labels):
        """Full UID averaging (100% UID mean, no individual weight).

        This is the aggressive version used in Kaggle submissions:
        Every transaction in a UID gets the same predicted probability.

        Args:
            predictions: numpy array of fraud probabilities.
            uid_labels: numpy array or Series of UID labels.

        Returns:
            numpy array of UID-averaged predictions.
        """
        df_temp = pd.DataFrame({'pred': predictions, 'uid': uid_labels})
        uid_means = df_temp.groupby('uid')['pred'].transform('mean')
        return uid_means.values
