import unittest

import numpy as np
import pandas as pd

from src.feature_engineering.view_behavioral import BehavioralExtractor
from src.feature_engineering.view_tabular import TabularFeatureExtractor
from src.feature_engineering.uid_features import UIDFeatureEngineer


class FeatureSafetyTests(unittest.TestCase):
    def test_behavioral_counts_are_causal(self):
        df = pd.DataFrame(
            {
                "card1": [1, 1, 2, 1],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            }
        )

        result = BehavioralExtractor.engineer_velocity(df)

        self.assertEqual(result["Card_Tx_Count"].tolist(), [1, 2, 1, 3])
        self.assertEqual(result["Card_Prior_Txn_Count"].tolist(), [0, 1, 0, 2])

    def test_categorical_encoder_does_not_learn_holdout_categories(self):
        df = pd.DataFrame({"kind": ["A", "B", "C", "A"]})

        result = TabularFeatureExtractor.encode_categoricals(df, fit_idx=np.array([0, 1]))

        self.assertEqual(result.loc[0, "kind"], 0)
        self.assertEqual(result.loc[1, "kind"], 1)
        self.assertEqual(result.loc[2, "kind"], -1)

    def test_nan_cleanup_uses_fit_slice_only(self):
        df = pd.DataFrame(
            {
                "keep": [1.0, 2.0, 3.0, 4.0],
                "drop_even_if_holdout_populated": [np.nan, np.nan, 9.0, 10.0],
                "UID": ["a", "b", "c", "d"],
            }
        )

        result = TabularFeatureExtractor.clean_high_nan_columns(
            df,
            threshold=0.7,
            fit_idx=np.array([0, 1]),
            preserve_cols=["UID"],
        )

        self.assertIn("keep", result.columns)
        self.assertIn("UID", result.columns)
        self.assertNotIn("drop_even_if_holdout_populated", result.columns)

    def test_v_pca_fits_without_holdout_distribution(self):
        df = pd.DataFrame(
            {
                "V1": [1.0, 2.0, 100.0, 101.0],
                "V2": [2.0, 3.0, 200.0, 201.0],
                "V3": [3.0, 4.0, 300.0, 301.0],
                "V4": [4.0, 5.0, 400.0, 401.0],
                "V5": [5.0, 6.0, 500.0, 501.0],
                "V6": [6.0, 7.0, 600.0, 601.0],
                "V7": [7.0, 8.0, 700.0, 701.0],
                "V8": [8.0, 9.0, 800.0, 801.0],
                "V9": [9.0, 10.0, 900.0, 901.0],
                "V10": [10.0, 11.0, 1000.0, 1001.0],
            }
        )

        result = UIDFeatureEngineer.v_column_pca(df, n_components=2, fit_idx=np.array([0, 1]))

        self.assertTrue(any(col.startswith("V_PCA_") for col in result.columns))
        self.assertFalse(any(col in result.columns for col in [f"V{i}" for i in range(1, 11)]))


if __name__ == "__main__":
    unittest.main()
