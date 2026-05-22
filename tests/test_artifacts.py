import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main_train_pipeline import save_holdout_artifacts


class ArtifactTests(unittest.TestCase):
    def test_holdout_artifacts_can_use_custom_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            holdout = pd.DataFrame(
                {
                    "TransactionID": [1, 2],
                    "UID": ["u1", "u2"],
                    "isFraud": [0, 1],
                }
            )

            save_holdout_artifacts(
                dataset="paysim",
                holdout_frame=holdout,
                fraud_scores=np.array([0.1, 0.9]),
                decisions=np.array(["ALLOW", "AUTO_BLOCK"]),
                artifacts_dir=tmp_dir,
            )

            output_path = Path(tmp_dir) / "paysim_holdout_predictions.csv"
            self.assertTrue(output_path.exists())
            saved = pd.read_csv(output_path)
            self.assertEqual(saved["decision"].tolist(), ["ALLOW", "AUTO_BLOCK"])


if __name__ == "__main__":
    unittest.main()
