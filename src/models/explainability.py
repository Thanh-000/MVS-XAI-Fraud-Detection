import shap
import matplotlib.pyplot as plt
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_xgboost_global_shap(model, X_train_sample: pd.DataFrame):
    """
    Vẽ biểu đồ Giải thích tính năng toàn cục (Global SHAP Summary Plot).
    Cho thấy Feature nào đang thao túng quyết định Fraud của cây quyết định.
    
    Args:
        model: Mô hình XGBClassifier đã train.
        X_train_sample: Một lượng mẫu train nhỏ (vd: 5000 dòng) để tiết kiệm thời gian tính Shap Values.
    """
    logger.info("Tính toán cơ sở SHAP Explainer cho dữ liệu Tabular...")
    
    # Sử dụng TreeExplainer (siêu tối ưu tốc độ cho XGBoost / LightGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sample)
    
    plt.figure(figsize=(10, 8))
    plt.title("Biểu đồ SHAP tóm tắt tầm quan trọng của các Features (Global MVS-XAI)", pad=20)
    shap.summary_plot(shap_values, X_train_sample, show=False)
    plt.tight_layout()
    plt.show()

def evaluate_xgboost_local_shap(model, transaction: pd.DataFrame, transaction_index=0):
    """
    Giải thích XAI ngay tại 1 giao dịch cụ thể định chỉ định (Local Explanation).
    Hiển thị biểu đồ dạng Lực đánh (Force Plot) hoặc Waterfall.
    
    Args:
        model: Cây XGB.
        transaction: DataFrame chứa giao dịch mục tiêu.
        transaction_index: Dòng số bao nhiêu trong DataFrame muốn tra tấn.
    """
    logger.info(f"Tiến hành kiểm toán chéo (Local SHAPX) cho giao dịch thứ {transaction_index}...")
    explainer = shap.Explainer(model)
    shap_values = explainer(transaction)
    
    # Thường được sử dụng trong Jupyter Notebooks để render HTML đồ họa tương tác
    shap.initjs()
    # Nếu chạy standalone script, ta sẽ vẽ qua Waterfall plot thay vì JS Force Plot
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[transaction_index], max_display=10, show=True)
