"""
TRUNG TÂM KIẾN TRÚC MỚI THEO CHUẨN ĐỒ ÁN (PaySim & IEEE-CIS Edition)
Điều hướng qua: TripleView -> KMeansSMOTE -> LSTM/MVS Stacking -> Evaluate -> XAI
"""
import sys
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.data.feature_engineering import TripleViewEngineer
from src.models.mvs_stacking import AdvancedMetaStacking
from src.models.lstm_network import SequentialTrainer
from src.models.evaluation import ModelEvaluator
from sklearn.model_selection import train_test_split

def boot_advanced_system():
    print("🚀 Bắt đầu Khởi chạy Đường Hầm Tự Động Kỷ Nguyên Mới Học Máy...\n")
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN 1: MOCK DATA CHO MỘT BỘ NHỎ ĐỂ CHẠY TEST KHÔNG OVERFLOW RAM (BẠN SẼ CẮM FILE THẬT Ở ĐÂY LÀ ĐƯỢC)
    # -------------------------------------------------------------
    import random
    print("Mô Phỏng Giao dịch PaySim Nhanh...")
    df = pd.DataFrame({
        'amount': np.random.rand(1000) * 5000,
        'oldbalanceOrg': np.random.rand(1000) * 10000,
        'newbalanceOrig': np.random.rand(1000) * 10000,
        'oldbalanceDest': np.random.rand(1000) * 50000,
        'newbalanceDest': np.random.rand(1000) * 50000,
        'nameOrig': [f'C{random.randint(1,50)}' for _ in range(1000)],
        'nameDest': [f'M{random.randint(1,50)}' for _ in range(1000)],
        'type': ['TRANSFER' for _ in range(1000)],
        'target': np.random.randint(0, 2, size=1000)
    })
    
    # Ép hacker lộ diện
    df.loc[42, 'target'] = 1
    
    # Triple-View
    X_features, y_target = TripleViewEngineer.process_paysim(df)
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN 2: CHIA CẮT VÀ ÉP SMOTE NHÂN CÂY (TÔN TRỌNG TẬP TEST)
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target, test_size=0.15, random_state=42, stratify=y_target
    )
    
    from imblearn.over_sampling import KMeansSMOTE, SMOTE
    try:
        smote = KMeansSMOTE(cluster_balance_threshold=0.01, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except Exception:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        
    print(f"✅ Bơm xong SMOTE: Tập Train Mồi nay có kích cỡ: {X_res.shape}")
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN 3+4: HUẤN LUYỆN OOF MATRIX (STACKING 4 TREES + 1 LSTM)
    # -------------------------------------------------------------
    stacker = AdvancedMetaStacking()
    # 1. Quét Cây Khủng long
    oof_matrix, val_labels = stacker.build_oof(X_res, y_res)
    
    # 2. Quét Mạng LSTM Sequential
    lstm_trainer = SequentialTrainer(in_features=X_res.shape[1])
    lstm_oof = lstm_trainer.train_and_predict_oof(X_res, y_res)
    
    # Ghép 5 thuật toán thành Tòa Án
    oof_final = np.column_stack((oof_matrix, lstm_oof))
    stacker.fit_meta(oof_final, val_labels)
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN KHỐC LIỆT: DỘI MODEL XUỐNG TEST HOLDOUT SẠCH CHƯA SMOTE
    # -------------------------------------------------------------
    stacker.refit_all(X_res, y_res)
    
    test_oof_trees = stacker.build_test_oof(X_test)
    test_oof_lstm = lstm_trainer.predict(X_test)
    test_oof_final = np.column_stack((test_oof_trees, test_oof_lstm))
    
    y_pred, y_probs = stacker.predict_meta(test_oof_final)
    
    # -------------------------------------------------------------
    # GIAI ĐOẠN ĐÁNH GIÁ (EVAL)
    # -------------------------------------------------------------
    from sklearn.metrics import classification_report, roc_auc_score
    print("\n================ BÁO CÁO CỰC ĐOAN TRÊN TẬP TEST GỐC (HOLDOUT) ================")
    print(f"ROC-AUC Toàn Năng: {roc_auc_score(y_test, y_probs):.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    
    print("\n>>> Module Model Đã Thành Công. Hệ thống chuẩn bị ném Model (XGB) về Tầng XAI...")

if __name__ == "__main__":
    boot_advanced_system()
