import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns in PaySim. 'type' is the main one.
    Use One-Hot Encoding for 'type'.
    """
    logger.info("Encoding categorical variables ('type')...")
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
    return df

def generate_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate tabular features as described in the MVS-XAI architecture.
    Calculates balance deltas, transaction amount vs balance ratios, etc.
    """
    logger.info("Generating tabular engineered features...")
    df_feat = df.copy()
    
    # Error in merchant/customer balances
    # If old - new + amount != 0, it means there is a hidden transfer or error
    df_feat['orig_balance_delta'] = df_feat['oldbalanceOrg'] - df_feat['newbalanceOrig'] - df_feat['amount']
    df_feat['dest_balance_delta'] = df_feat['oldbalanceDest'] - df_feat['newbalanceDest'] + df_feat['amount']
    
    # Ratios (adding small epsilon to prevent division by zero)
    eps = 1e-6
    df_feat['amount_to_oldbalanceOrg_ratio'] = df_feat['amount'] / (df_feat['oldbalanceOrg'] + eps)
    df_feat['amount_to_oldbalanceDest_ratio'] = df_feat['amount'] / (df_feat['oldbalanceDest'] + eps)
    
    # Log transformation for highly skewed amount
    df_feat['log_amount'] = np.log1p(df_feat['amount'])
    
    # Note: nameOrig and nameDest strings usually indicate Merchant vs Customer (M vs C)
    # E.g., 'M12345'
    if 'nameDest' in df_feat.columns:
        df_feat['is_dest_merchant'] = df_feat['nameDest'].astype(str).str.startswith('M').astype(np.int8)
    
    # Drop identifiers and unused features
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    cols_to_drop = [c for c in cols_to_drop if c in df_feat.columns]
    df_feat.drop(columns=cols_to_drop, inplace=True)
    
    return df_feat

def preprocess_pipeline(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies the full preprocessing pipeline across train, val, and test splits.
    Ensures that any fit happens ONLY on the training set to prevent data leakage.
    (e.g., standard scalers would be fitted here later)
    """
    logger.info("Starting preprocessing pipeline on temporal splits...")
    
    # Generate features
    train = generate_tabular_features(train)
    val = generate_tabular_features(val)
    test = generate_tabular_features(test)
    
    # Encode categorical
    train = encode_categorical(train)
    val = encode_categorical(val)
    test = encode_categorical(test)
    
    # Ensure same columns are present in all sets (in case some categories were missing in splits)
    common_cols = list(train.columns.intersection(val.columns).intersection(test.columns))
    
    # Important: Keep the label columns
    if 'isFraud' not in common_cols:
        common_cols.append('isFraud')
        
    return train[common_cols], val[common_cols], test[common_cols]
