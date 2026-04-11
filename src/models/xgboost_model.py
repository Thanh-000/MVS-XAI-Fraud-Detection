import pandas as pd
import numpy as np
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

def evaluate_scale_pos_weight(y_train: np.ndarray) -> float:
    """ Tính toán trọng số tự cân bằng nội bộ của XGBoost. """
    fraud_count = np.sum(y_train)
    if fraud_count == 0:
        return 1.0
    return (len(y_train) - fraud_count) / float(fraud_count)

def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, max_iters=500):
    """
    XGBoost Tabular Detector for Fraud Anomaly Detection.
    Tính toán Scale Pos Weight khổng lồ (Do imbalance ratio 1:1000 trong PaySim).
    Targeting PR-AUC rather than ROC-AUC due to class skewness.
    """
    logger.info("⚡ Khởi động hệ thống đào tạo lõi XGBoost Tabular Detector...")
    
    weight = evaluate_scale_pos_weight(y_train)
    logger.info(f"Áp dụng Class Balance Weight = {weight:.2f}")
    
    # Thiết lập sức mạnh lõi
    clf = xgb.XGBClassifier(
        n_estimators=max_iters,
        max_depth=6,             # Tree không quá sâu để tránh Overfit vào noise
        learning_rate=0.03,      # Căn chỉnh bước nhảy an toàn
        scale_pos_weight=weight, # Bắt trọn Fraud thay vì phớt lờ
        tree_method='hist',      # Xử lý ma trận dữ liệu hàng triệu dòng
        eval_metric='aucpr',     # Metrics chuyên dụng cho Imbalanced Data (Precision-Recall)
        early_stopping_rounds=30,# Chìa khóa chống Overfit tự động
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("⏳ Chạy huấn luyện (Quá trình này có thể kéo dài phụ thuộc vào tài nguyên phần cứng)...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25
    )
    
    logger.info(f"✅ Đào tạo kết thúc. Best Score tại Vòng [{clf.best_iteration}]: {clf.best_score:.4f}")
    
    return clf
