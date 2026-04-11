"""
Liên minh Cây Khủng Long (Base Models) & Tòa Án Tối Cao (Logistic Regression Meta Learner)
Giải Quyết Bài toán Cố hữu Bằng Out-of-Fold Matrix (Hạn chế hoàn toàn Data Leakage)
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np

class AdvancedMetaStacking:
    def __init__(self, random_state=42):
        self.rs = random_state
        print("[META MVS] Đang đúc kết Bộ tứ Cây Quyết định cực đoạn (RF, XGB, LightGBM, CatBoost)...")
        self.models = {
            'RF': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=self.rs),
            'XGB': XGBClassifier(n_estimators=100, max_depth=6, tree_method='hist', random_state=self.rs),
            'LGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=self.rs, verbose=-1),
            'CAT': cat.CatBoostClassifier(n_estimators=100, depth=6, verbose=0, random_state=self.rs)
        }
        self.meta_lr = None
        
    def apply_gating(self, matrix):
        """Khử nhiễu các suy đoán lấp lửng (0.4 - 0.6) của các Base Models"""
        mask = np.where((matrix > 0.4) & (matrix < 0.6), 0.5, 1.0)
        return matrix * mask

    def build_oof(self, X, y):
        print("[META MVS] Thiết lập Ma trận Bẻ gãy Rò rỉ OOF (Out-Of-Fold) qua K-Fold Cross Validation...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.rs)
        oof_matrix = np.zeros((len(X), len(self.models)))
        val_labels = np.zeros(len(X))
        
        # X & y đầu vào là DataFrame hoặc Series, đã gỡ Index để tránh Lỗi Loc
        X_val = X.values if hasattr(X, 'values') else X
        y_val_list = y.values if hasattr(y, 'values') else y
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_val, y_val_list)):
            X_trn, y_trn = X_val[trn_idx], y_val_list[trn_idx]
            X_val_split, y_val_split = X_val[val_idx], y_val_list[val_idx]
            val_labels[val_idx] = y_val_split
            
            for i, (name, clf) in enumerate(self.models.items()):
                clf.fit(X_trn, y_trn)
                oof_matrix[val_idx, i] = clf.predict_proba(X_val_split)[:, 1]
                
        return oof_matrix, val_labels
        
    def refit_all(self, X, y):
        """Re-fit full data để đón đánh tập Kiểm định (Test Hold-out)"""
        print("[META MVS] Giải nén toàn bộ giới hạn công suất trên dữ liệu Full Train (Chiến Test)...")
        for name, clf in self.models.items():
            clf.fit(X, y)
            
    def build_test_oof(self, X_test):
        test_oof = np.zeros((len(X_test), len(self.models)))
        for i, (name, clf) in enumerate(self.models.items()):
            test_oof[:, i] = clf.predict_proba(X_test)[:, 1]
        return test_oof
        
    def fit_meta(self, oof_matrix, val_labels):
        print("=> Gắn ngòi nổi Trọng Tài Tối cao (Meta Learner)...")
        gated = self.apply_gating(oof_matrix)
        self.meta_lr = LogisticRegression(class_weight='balanced', C=0.1)
        self.meta_lr.fit(gated, val_labels)
        
    def predict_meta(self, test_oof_matrix):
        gated = self.apply_gating(test_oof_matrix)
        return self.meta_lr.predict(gated), self.meta_lr.predict_proba(gated)[:, 1]
