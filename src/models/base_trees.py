"""
Tree Ensemble Factory — Hyperparameters matching notebook 06 (v4.3.4).

All 4 tree models use the EXACT same hyperparameters as the research notebook:
- XGBoost: 800 rounds, depth=8, lr=0.03, early_stopping=50
- LightGBM: 800 rounds, depth=8, lr=0.03, early_stopping via callback
- CatBoost: 800 iterations, depth=8, lr=0.03, early_stopping=50
- RandomForest: 500 trees, depth=15, min_samples_leaf=10

Imbalance handling: SMOTE+CTGAN (external), NOT native class weights.
"""
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


class TreeEnsembleFactory:
    """Factory for creating tree-based ensemble models with research-grade hyperparameters."""

    @staticmethod
    def get_random_forest(seed=42, profile="research"):
        """RandomForest: 500 trees, max_depth=15.
        No native class weights — imbalance handled by SMOTE+CTGAN externally.
        """
        if profile == "fast":
            n_estimators = 160
            max_depth = 12
        else:
            n_estimators = 500
            max_depth = 15
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            class_weight=None,   # SMOTE handles imbalance
            n_jobs=-1,
            random_state=seed
        )

    @staticmethod
    def get_xgboost(use_gpu=True, seed=42, profile="research"):
        """XGBoost: 800 rounds, lr=0.03, depth=8, early_stopping=50.
        GPU-accelerated via hist tree method.
        """
        if XGBClassifier is None:
            raise ImportError("XGBoost not installed: pip install xgboost")
        if profile == "fast":
            n_estimators = 360
            learning_rate = 0.05
            max_depth = 6
            early_stopping_rounds = 30
        else:
            n_estimators = 800
            learning_rate = 0.03
            max_depth = 8
            early_stopping_rounds = 50
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            tree_method='hist',
            scale_pos_weight=1.0,  # NEUTRAL — SMOTE handles imbalance
            gamma=2,
            reg_alpha=0.5,
            reg_lambda=2,
            early_stopping_rounds=early_stopping_rounds,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=seed,
            verbosity=0,
        )
        if use_gpu:
            params['device'] = 'cuda'
        return XGBClassifier(**params)

    @staticmethod
    def get_lightgbm(seed=42, profile="research"):
        """LightGBM: 800 rounds, lr=0.03, depth=8, num_leaves=63.
        Early stopping handled via lgb.early_stopping callback during fit().
        """
        if LGBMClassifier is None:
            raise ImportError("LightGBM not installed: pip install lightgbm")
        if profile == "fast":
            n_estimators = 420
            learning_rate = 0.05
            max_depth = 7
            num_leaves = 47
        else:
            n_estimators = 800
            learning_rate = 0.03
            max_depth = 8
            num_leaves = 63
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=100,
            is_unbalance=False,  # DISABLED — SMOTE handles imbalance
            subsample=0.8,
            colsample_bytree=0.7,
            n_jobs=-1,
            random_state=seed,
            verbose=-1
        )

    @staticmethod
    def get_catboost(use_gpu=True, seed=42, profile="research"):
        """CatBoost: 800 iterations, lr=0.03, depth=8, early_stopping=50."""
        if CatBoostClassifier is None:
            raise ImportError("CatBoost not installed: pip install catboost")
        if profile == "fast":
            iterations = 360
            learning_rate = 0.05
            depth = 6
            early_stopping_rounds = 30
        else:
            iterations = 800
            learning_rate = 0.03
            depth = 8
            early_stopping_rounds = 50
        params = dict(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            auto_class_weights=None,  # REMOVED — SMOTE handles imbalance
            early_stopping_rounds=early_stopping_rounds,
            verbose=0,
            random_state=seed
        )
        if use_gpu:
            params['task_type'] = 'GPU'
        return CatBoostClassifier(**params)
