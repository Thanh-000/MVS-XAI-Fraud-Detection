"""
Khai khoáng Dữ liệu (Feature Engineering)
Quản lý Góc Phân Tích Đa Chiều (Triple-View): Tabular, Behavioral và Sequential.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TripleViewEngineer:
    @staticmethod
    def process_paysim(df):
        print("💡 [TRIPLE-VIEW] Kích hoạt Trích xuất Behavior & Error Kế toán (PAYSIM)...")
        df_tab = df.copy()
        
        # TABULAR VIEW
        df_tab['ErrorBalanceOrig'] = df_tab['newbalanceOrig'] + df_tab['amount'] - df_tab['oldbalanceOrg']
        df_tab['ErrorBalanceDest'] = df_tab['oldbalanceDest'] + df_tab['amount'] - df_tab['newbalanceDest']

        for c in ['type', 'nameOrig', 'nameDest']:
            if c in df_tab.columns:
                df_tab[c] = LabelEncoder().fit_transform(df_tab[c].astype(str))
                
        # BEHAVIORAL VIEW
        df_tab['User_Velocity'] = df_tab.groupby('nameOrig')['amount'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df_tab['User_Txn_Count'] = df_tab.groupby('nameOrig').cumcount() + 1
        
        df_tab.fillna(-999, inplace=True)
        y = df_tab['target'].values
        X = df_tab.drop(columns=['target'])
        return X, y

    @staticmethod
    def process_ieee(df, df_id=None):
        print("💡 [TRIPLE-VIEW] Kích hoạt Trích xuất Thẻ Ẩn Danh (IEEE-CIS)...")
        raw_df = df if df_id is None else df.merge(df_id, on='TransactionID', how='left')
        if 'isFraud' in raw_df.columns: 
            raw_df.rename(columns={'isFraud':'target'}, inplace=True)
            
        raw_df.fillna(-999, inplace=True)
        
        # Lọc các trường thiết yếu để nhẹ RAM
        keep_cols = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'dist1'] \
                    + [f'C{i}' for i in range(1, 15)] + [f'V{i}' for i in range(300, 310)]
        df_tab = raw_df[[c for c in keep_cols if c in raw_df.columns] + ['target']].copy()
        
        for c in df_tab.select_dtypes(include=['object']).columns:
            df_tab[c] = LabelEncoder().fit_transform(df_tab[c].astype(str))
            
        # BEHAVIORAL VIEW
        df_tab['Card_Velocity'] = df_tab.groupby('card1')['TransactionAmt'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df_tab['Card_Freq'] = df_tab.groupby('card1').cumcount() + 1
        
        df_tab.fillna(-999, inplace=True)
        y = df_tab['target'].values
        X = df_tab.drop(columns=['target'])
        return X, y
