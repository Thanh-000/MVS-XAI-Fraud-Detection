"""
Khối Đánh giá Mô hình (Metrics & Evaluation)
Thống kê Classification Report, ROC-AUC mượn từ Colab.
"""
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test, model_name="MVS Ensemble"):
        print(f"\n================ ĐÁNH GIÁ MÔ HÌNH: {model_name} ================")
        
        # Dự đoán
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        print("[METRICS] Báo cáo Phân loại (Classification Report):")
        print(classification_report(y_test, y_pred, digits=4))
        
        auc = roc_auc_score(y_test, y_proba)
        print(f">>> Độ phủ ROC-AUC Score: {auc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f">>> Ma trận nhầm lẫn (Confusion Matrix):\n{cm}")
        
        # Trả về kết quả để XAI dùng định vị True Positive
        return y_pred, y_proba
