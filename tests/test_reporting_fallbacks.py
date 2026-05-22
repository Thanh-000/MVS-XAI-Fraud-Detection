import unittest

import pandas as pd

from src.evaluation.fairness import FairnessAuditor
from src.evaluation.wasserstein import DriftDetector


class ReportingFallbackTests(unittest.TestCase):
    def test_wasserstein_table_print_falls_back_without_tabulate(self):
        table = pd.DataFrame({"Feature": ["x"], "Wasserstein_Distance": [1.0], "Status": ["DRIFT"]})

        DriftDetector._print_table(table)

    def test_fairness_table_print_falls_back_without_tabulate(self):
        table = pd.DataFrame({"Group": ["A"], "Size": [1], "Block Rate": ["0.00%"]})

        FairnessAuditor._print_table(table)


if __name__ == "__main__":
    unittest.main()
