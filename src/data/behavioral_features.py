import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng Hành vi (Behavioral Velocity).
    Lưu ý quan trọng trên PaySim: 
    Trong PaySim, hầu hết nameOrig chỉ giao dịch 1-2 lần.
    Tuy nhiên, nameDest (Người nhận/Merchant) có mức độ tập trung giao dịch cực cao,
    đặc biệt là mục tiêu của mạng lưới lừa đảo (chuyển tiền/Cash Out).
    Do đó, View Hành Vi này sẽ xoáy vào nameDest.
    """
    logger.info("Tiến hành trích xuất Behavioral (Velocity/History) Features trên nameDest...")
    df_f = df.copy()
    
    # Bảo đảm tính Walk-Forward (sắp xếp theo thời gian)
    df_f = df_f.sort_values(by=['step']).reset_index(drop=True)
    
    # 1. Lifetime Transaction Count (Mật độ giao dịch tích luỹ của Người Nhận)
    # Lệnh này đếm hiện tại là giao dịch thứ mấy mà Dest này nhận được.
    logger.info("Tính toán dest_lifetime_txn_count...")
    df_f['dest_lifetime_txn_count'] = df_f.groupby('nameDest').cumcount() + 1
    
    # 2. Bước nhảy thời gian (Khoảng cách giờ giữa 2 giao dịch liên tiếp của cùng 1 Dest)
    logger.info("Tính toán chu kỳ thời gian (hours_since_last_txn)...")
    # Shift step theo từng NameDest để tìm bước trước đó
    df_f['dest_prev_step'] = df_f.groupby('nameDest')['step'].shift(1)
    df_f['dest_hours_since_last_txn'] = (df_f['step'] - df_f['dest_prev_step']).fillna(999) # 999 là giá trị đánh dấu giao dịch đầu tiên
    
    # 3. Lượng tiền trung bình nhận được trước giao dịch hiện tại (Trình tự thời gian thực)
    # Ta phải sử dụng expanding mean shift(1) để KHÔNG lấy chính lượng tiền hiện tại vào trung bình quá khứ
    logger.info("Tính toán dest_historical_avg_amount...")
    df_f['dest_historical_avg_amount'] = (
        df_f.groupby('nameDest')['amount']
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0)
    )
    
    # Dọn dẹp cột tạm
    df_f.drop(columns=['dest_prev_step'], inplace=True)
    
    return df_f

def create_lstm_sequences(df: pd.DataFrame, sequence_length: int = 10, feature_cols: list = None):
    """
    Tạo ra chuỗi (Sequences) dạng 3D tensor cho LSTM model.
    Shape mục tiêu: (batch_size, sequence_length, num_features).
    Trong mô hình PaySim, ta sẽ tạo sequences dựa theo dòng chảy tiền (các giao dịch nối tiếp vào hệ thống)
    để LSTM bắt được 'Global Anomaly Pattern' thay vì chỉ User Pattern (vì User chỉ gd 1 lần).
    """
    logger.info(f"Đang sinh LSTM Window (T={sequence_length})...")
    
    if feature_cols is None:
        # Tự động loại bỏ các cọc chữ hoặc label khỏi cụm Feature
        exclude = ['step', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 'type']
        feature_cols = [c for c in df.columns if c not in exclude]
        
    data = df[feature_cols].values
    labels = df['isFraud'].values if 'isFraud' in df.columns else None
    
    n_samples = len(data)
    X_seq = []
    y_seq = []
    
    # Padding cho các step đầu tiên không đủ sequence
    # Thay vì loại bỏ, ta zero-pad ở phía bên trái (theo đúng chuẩn time-series).
    # Việc thực hiện numpy padding trước sẽ làm sliding window chạy siêu nhanh (mất vài giây thay vì vài tiếng).
    
    pad_len = sequence_length - 1
    padded_data = np.pad(data, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Vectorized Sliding Window using lib stridelinks for maximum performance on large Numpy Arrays
    shape = (n_samples, sequence_length, data.shape[1])
    strides = (padded_data.strides[0], padded_data.strides[0], padded_data.strides[1])
    X_seq = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
    
    if labels is not None:
        y_seq = labels # Nhãn của giao dịch hiện tại ở cuối Sequence
        return X_seq, y_seq
    
    return X_seq
