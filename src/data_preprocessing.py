import pandas as pd
import numpy as np
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_test_split_users(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(random_state)
    users = df['user_id'].unique()
    rng.shuffle(users)
    split = int(len(users) * (1 - test_size))
    train_users = set(users[:split])
    train_df = df[df['user_id'].isin(train_users)].copy()
    test_df = df[~df['user_id'].isin(train_users)].copy()
    return train_df, test_df
