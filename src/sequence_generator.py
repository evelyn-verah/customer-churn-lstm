import numpy as np
import pandas as pd
from typing import Tuple

def build_user_sequences(df: pd.DataFrame, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = ['txn_count', 'cashin', 'cashout', 'failed_login', 'pin_reset']
    X_list, y_list = [], []
    for user_id, grp in df.groupby('user_id'):
        grp = grp.sort_values('day')
        feats = grp[feature_cols].values
        label = grp['churned'].iloc[0]
        if len(feats) >= seq_len:
            seq = feats[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(feats), feats.shape[1]))
            seq = np.vstack([pad, feats])
        X_list.append(seq)
        y_list.append(label)
    return np.stack(X_list), np.array(y_list)
