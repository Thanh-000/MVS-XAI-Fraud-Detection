import unittest

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


if __name__ == "__main__":
    unittest.main()
