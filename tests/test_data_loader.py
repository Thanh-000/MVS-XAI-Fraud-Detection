import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_pipeline.data_loader import DataLoader


class DataLoaderTests(unittest.TestCase):
    def test_reduce_mem_usage_skips_string_dtypes(self):
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0],
                "label": pd.Series(["PAYMENT", "TRANSFER", "CASH_OUT"], dtype="string"),
            }
        )

        reduced = DataLoader._reduce_mem_usage(df)

        self.assertIn("label", reduced.columns)
        self.assertEqual(reduced["label"].tolist(), ["PAYMENT", "TRANSFER", "CASH_OUT"])

    def test_resolve_paysim_path_finds_schema_in_nested_csv_with_unknown_name(self):
        with TemporaryDirectory() as tmp_dir:
            nested = Path(tmp_dir) / "extracted"
            nested.mkdir()
            csv_path = nested / "paysim dataset.csv"
            pd.DataFrame(
                {
                    "step": [1],
                    "type": ["TRANSFER"],
                    "amount": [100.0],
                    "nameOrig": ["C1"],
                    "nameDest": ["M1"],
                    "oldbalanceOrg": [100.0],
                    "newbalanceOrig": [0.0],
                    "oldbalanceDest": [0.0],
                    "newbalanceDest": [100.0],
                    "isFraud": [1],
                }
            ).to_csv(csv_path, index=False)

            resolved = DataLoader._resolve_paysim_path(tmp_dir)

            self.assertEqual(Path(resolved), csv_path)


if __name__ == "__main__":
    unittest.main()
