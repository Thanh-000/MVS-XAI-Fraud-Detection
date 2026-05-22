import unittest

import numpy as np

import main_train_pipeline as pipeline
from src.data_pipeline.data_sampler import DataBalanceEngine


class RuntimeFallbackTests(unittest.TestCase):
    def test_tree_preset_excludes_neural_branches(self):
        active = pipeline.get_active_model_names("cpu", seed=42, preset="tree")

        self.assertNotIn("MLP", active)
        self.assertNotIn("LSTM", active)
        self.assertIn("RF", active)

    def test_auto_preset_skips_neural_when_torch_missing(self):
        active = pipeline.get_active_model_names("cpu", seed=42, preset="auto")

        if not pipeline.module_available("torch"):
            self.assertNotIn("MLP", active)
            self.assertNotIn("LSTM", active)

    def test_fast_mvs_disables_lstm(self):
        active = pipeline.get_active_model_names("cpu", seed=42, preset="fast_mvs")

        self.assertNotIn("LSTM", active)
        if pipeline.module_available("torch"):
            self.assertIn("MLP", active)
        self.assertEqual(pipeline.resolve_model_profile("fast_mvs"), "fast")

    def test_unique_entity_sequences_are_lazy_zero_tensors(self):
        X_train = np.ones((4, 3), dtype=np.float32)
        X_val = np.ones((2, 3), dtype=np.float32)
        train_seq, val_seq = pipeline.build_sequence_train_val(
            X_train,
            X_val,
            entity_train=np.array([1, 2, 3, 4]),
            entity_val=np.array([5, 6]),
        )

        self.assertTrue(getattr(train_seq, "is_zero_sequence", False))
        self.assertTrue(getattr(val_seq, "is_zero_sequence", False))
        self.assertEqual(train_seq.shape, (4, 10, 3))
        self.assertEqual(val_seq[:1].shape, (1, 10, 3))

    def test_resampling_disabled_returns_original_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1])

        X_out, y_out = pipeline.prepare_tree_training_data(
            X,
            y,
            feature_names=["x"],
            device_type="cpu",
            seed=42,
            smote_strategy=0,
            ctgan_samples=0,
        )

        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_sampler_imports_without_optional_dependencies(self):
        engine = DataBalanceEngine(random_state=42)

        self.assertEqual(engine.random_state, 42)


if __name__ == "__main__":
    unittest.main()
