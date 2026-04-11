"""
Khối Xử lý Dữ liệu (Data Preprocessing Pipeline)
Code chuẩn hóa từ các ô Colab: Nạp CSV, Xử lý Missing, Encoding, Tách tập SMOTE.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class DataPipeline:
    def __init__(self, filepath, random_state=42):
        self.filepath = filepath
        self.rs = random_state
        self.scaler = StandardScaler()
        
    def load_and_preprocess(self):
        """Khôi phục quy trình xử lý thực tế trên Colab"""
        print(f"[PREPROCESSING] Đang tải file CSV từ: {self.filepath}")
        try:
            df = pd.read_csv(self.filepath)
            print(f"[PREPROCESSING] Load thành công Dataset. Shape: {df.shape}")
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file {self.filepath}. Bạn cần tải CSV từ Kaggle vào mục data/ nhé.")
            return None, None
            
        # Tìm Cột Target
        target_col = 'Class' if 'Class' in df.columns else 'isFraud'
        if target_col not in df.columns:
            target_col = df.columns[-1]
            
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        print("[PREPROCESSING] Đang điền khuyết Missing Values (Mô phỏng Colab)...")
        # Xử lý NaN cơ bản dựa theo Colab (Điền -999 hoặc Mean)
        X.fillna(-999, inplace=True)
        
        # Xử lý category
        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print(f"[PREPROCESSING] Đang mã hóa biến Categorical: {list(cat_cols)}")
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            
        return X, y
        
    def split_and_balance(self, X, y, test_size=0.2):
        """Chia tập và Bơm SMOTE"""
        print("[TRAIN/TEST] Tách tập Validation...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.rs
        )
        
        print("[SMOTE] Đang mở rộng dữ liệu thiểu số (Lừa đảo) với công nghệ SMOTE...")
        smote = SMOTE(random_state=self.rs)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Hoàn tất. Size Train trước cân bằng: {len(y_train)}, sau cân bằng: {len(y_train_res)}")
        
        return X_train_res, X_test, y_train_res, y_test
