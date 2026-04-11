import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_paysim(data_dir: str, filename: str = "PS_20174392719_1491204439457_log.csv") -> pd.DataFrame:
    """
    Loads the PaySim dataset from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing the dataset.
        filename (str): Name of the PaySim CSV file.
        
    Returns:
        pd.DataFrame: PaySim dataframe.
    """
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Dataset not found at {file_path}. Please download it from Kaggle.")
        raise FileNotFoundError(f"Missing file: {file_path}")
        
    logger.info(f"Loading PaySim dataset from {file_path}...")
    
    # Load dataset. We optimize memory types to prevent RAM issues on 6.3M rows
    dtypes = {
        'step': 'int16',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'category',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'category',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    df = pd.read_csv(file_path, dtype=dtypes)
    
    logger.info(f"Loaded {len(df):,} samples.")
    logger.info(f"Fraud distribution:\n{df['isFraud'].value_counts(normalize=True)*100}")
    
    return df

def create_temporal_splits(df: pd.DataFrame, time_col: str = 'step', train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Creates temporal train/val/test splits to prevent future-snooping leakage.
    Ensures that validation data happens strictly after training data, 
    and testing data happens strictly after validation data.
    
    Args:
        df: Input DataFrame
        time_col: Column representing time (e.g., 'step' in PaySim is 1 hour unit)
        train_ratio: Float representing training share.
        val_ratio: Float representing validation share.
        
    Returns:
        train, val, test pd.DataFrame
    """
    logger.info(f"Creating temporal splits based on the column '{time_col}'...")
    
    # Needs to be sorted by time to prevent leakage
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()
    
    logger.info(f"Temporal Split Results:")
    logger.info(f"Train: {len(train):,} samples (Steps {train[time_col].min()} to {train[time_col].max()}) | Fraud: {train['isFraud'].sum()}")
    logger.info(f"Val:   {len(val):,} samples (Steps {val[time_col].min()} to {val[time_col].max()}) | Fraud: {val['isFraud'].sum()}")
    logger.info(f"Test:  {len(test):,} samples (Steps {test[time_col].min()} to {test[time_col].max()}) | Fraud: {test['isFraud'].sum()}")
    
    return train, val, test
