"""
View 3: Behavioral Feature Extraction — Rolling window velocity features.
Matches notebook 06 — 7/14/30-day rolling per card account.
"""
import pandas as pd
import numpy as np


class BehavioralExtractor:
    """Extract behavioral velocity features: rolling means, std, spending velocity."""

    @staticmethod
    def engineer_velocity(df):
        """Compute 7/14/30-day rolling velocity features per card1 account.

        Features created:
        - Card_Velocity_{7d,14d,30d}: Rolling mean of TransactionAmt
        - Card_Amt_Std_{7d,14d,30d}: Rolling std of TransactionAmt
        - Card_Spending_Velocity: Rolling count (transaction frequency)
        - Amt_Deviation: Z-score vs 7-day personal baseline
        - Amt_to_Mean_Ratio: Current amount / 30-day moving average
        - Card_Prior_Txn_Count: Cumulative transaction count (for cold-start detection)
        """
        print("  View 3 (Behavioral): 7/14/30-day rolling velocity features...")

        if df['card1'].nunique(dropna=False) == len(df):
            for label in ['7d', '14d', '30d']:
                df[f'Card_Velocity_{label}'] = df['TransactionAmt']
                df[f'Card_Amt_Std_{label}'] = 0.0
            df['Card_Tx_Count'] = 1
            df['Card_Spending_Velocity'] = 1.0
            df['Amt_Deviation'] = 0.0
            df['Amt_to_Mean_Ratio'] = df['TransactionAmt'] / (df['TransactionAmt'] + 1)
            df['Card_Prior_Txn_Count'] = 0
            print(f"    Unique card1 per row; used exact O(N) cold-start velocity path")
            print(f"    Cold-start accounts (<3 txns): {len(df):,} transactions")
            return df

        grouped_amount = df.groupby('card1', sort=False)['TransactionAmt']
        for window, label in [(7, '7d'), (14, '14d'), (30, '30d')]:
            df[f'Card_Velocity_{label}'] = (
                grouped_amount.rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f'Card_Amt_Std_{label}'] = (
                grouped_amount.rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

        # Spending velocity (count of txns observed so far). Do not use
        # transform('count'), which would reveal future transactions per card.
        prior_count = df.groupby('card1').cumcount()
        df['Card_Tx_Count'] = prior_count + 1
        df['Card_Spending_Velocity'] = (
            grouped_amount.rolling(10, min_periods=1)
            .count()
            .reset_index(level=0, drop=True)
        )

        # Deviation from personal baseline
        df['Amt_Deviation'] = (
            (df['TransactionAmt'] - df['Card_Velocity_7d']) / (df['Card_Amt_Std_7d'] + 1)
        )
        df['Amt_to_Mean_Ratio'] = df['TransactionAmt'] / (df['Card_Velocity_30d'] + 1)

        # Cold-start detection: cumulative count per card before this transaction
        df['Card_Prior_Txn_Count'] = prior_count
        n_cold = (df['Card_Prior_Txn_Count'] < 3).sum()
        print(f"    Cold-start accounts (<3 txns): {n_cold:,} transactions")

        return df
