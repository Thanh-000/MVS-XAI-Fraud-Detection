"""
HITL (Human-in-the-Loop) Router — 3-tier transaction routing.
Matches notebook 06 (v4.3.4).

Decision logic:
  - AUTO_BLOCK: fraud_score ≥ 0.60 (high confidence fraud)
  - HITL_REVIEW: score ∈ [0.35, 0.60) AND additional conditions
  - ALLOW: score < 0.35 (likely legitimate)

Refinements (v4.3.4):
  - High-value promotion: amount > 95th percentile AND score ≥ 0.35
  - Cold-start detection: Card_Prior_Txn_Count < 3 → HITL instead of ALLOW
  - New-client flag: is_new_client → route to HITL if score ≥ 0.20
"""
import numpy as np
import pandas as pd


class HITLRouter:
    """3-tier HITL transaction routing with fraud score thresholds."""

    def __init__(self, auto_block_threshold=0.60, review_threshold=0.35,
                 amount_percentile=95, cold_start_txn_limit=3):
        self.auto_block_threshold = auto_block_threshold
        self.review_threshold = review_threshold
        self.amount_percentile = amount_percentile
        self.cold_start_txn_limit = cold_start_txn_limit

    def route_transactions(self, df, fraud_scores, amounts=None,
                           prior_txn_counts=None, new_client_flags=None):
        """Route each transaction to AUTO_BLOCK, HITL_REVIEW, or ALLOW.

        Args:
            df: DataFrame with transaction data (for context).
            fraud_scores: Array of meta-learner fraud probabilities.
            amounts: Optional array of TransactionAmt (for high-value promotion).
            prior_txn_counts: Optional array of Card_Prior_Txn_Count.
            new_client_flags: Optional array of is_new_client (0/1).

        Returns:
            DataFrame with 'decision' column added.
        """
        decisions = np.full(len(fraud_scores), 'ALLOW', dtype=object)

        # 1. Base routing
        decisions[fraud_scores >= self.auto_block_threshold] = 'AUTO_BLOCK'
        mask_review = (fraud_scores >= self.review_threshold) & \
                      (fraud_scores < self.auto_block_threshold)
        decisions[mask_review] = 'HITL_REVIEW'

        # 2. High-value promotion
        if amounts is not None:
            amt_threshold = np.percentile(amounts, self.amount_percentile)
            high_value_promote = (amounts > amt_threshold) & \
                                 (fraud_scores >= self.review_threshold) & \
                                 (decisions == 'ALLOW')
            decisions[high_value_promote] = 'HITL_REVIEW'

        # 3. Cold-start detection
        if prior_txn_counts is not None:
            cold_start = (prior_txn_counts < self.cold_start_txn_limit) & \
                         (decisions == 'ALLOW') & \
                         (fraud_scores >= 0.20)
            decisions[cold_start] = 'HITL_REVIEW'

        # 4. New-client routing
        if new_client_flags is not None:
            new_client = (new_client_flags == 1) & \
                         (decisions == 'ALLOW') & \
                         (fraud_scores >= 0.20)
            decisions[new_client] = 'HITL_REVIEW'

        df = df.copy()
        df['fraud_score'] = fraud_scores
        df['decision'] = decisions

        # Report
        n_block = (decisions == 'AUTO_BLOCK').sum()
        n_review = (decisions == 'HITL_REVIEW').sum()
        n_allow = (decisions == 'ALLOW').sum()
        print(f"\n  HITL Routing Summary:")
        print(f"    AUTO_BLOCK:  {n_block:>8,} ({n_block/len(decisions)*100:5.2f}%)")
        print(f"    HITL_REVIEW: {n_review:>8,} ({n_review/len(decisions)*100:5.2f}%)")
        print(f"    ALLOW:       {n_allow:>8,} ({n_allow/len(decisions)*100:5.2f}%)")
        return df
