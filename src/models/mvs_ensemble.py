"""
Khối Thuật toán Lõi (Core ML Models)
Nơi khởi tạo Meta-Model / Bộ phân loại bỏ phiếu MVS (Multi-Model Voting System)
"""
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

class MVSEnsembleBuilder:
    def __init__(self, random_state=42):
        self.rs = random_state
        
    def build_ensemble(self):
        print("[MVS CORE] Cấu trúc Tường lửa Đa Phương Thức (XGB + RF + LR) đang được khởi tạo...")
        
        # 1. XGBoost (Tướng tiên phong bắt pattern dị thường phức tạp)
        xgb = XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=300,
            scale_pos_weight=10, eval_metric='logloss', random_state=self.rs
        )
        
        # 2. Random Forest (Lá chắn thứ hai chống Overfitting)
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=self.rs
        )
        
        # 3. Logistic Regression (Tòa án Toán học tuyến tính kéo quyết định bảo hiểm rủi ro)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        
        # 4. Meta-Ensemble (Sự hợp nhất mềm Cấp độ Probability)
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
            voting='soft'
        )
        
        return ensemble, xgb # Trả về ensemble để predict chính, trả về xgb cho XAI mổ xẻ
