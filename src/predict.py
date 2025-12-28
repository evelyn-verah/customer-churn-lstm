import pathlib
import pandas as pd
from tensorflow.keras.models import load_model

from data_preprocessing import load_data
from sequence_generator import build_user_sequences

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mobile_money_logs.csv"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "lstm_model.h5"
SEQ_LEN = 30

def main():
    df = load_data(str(DATA_PATH))
    X, y = build_user_sequences(df, seq_len=SEQ_LEN)
    user_ids = df['user_id'].unique()

    model = load_model(MODEL_PATH)
    y_prob = model.predict(X).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        'user_id': user_ids,
        'churn_probability': y_prob,
        'predicted_churn': y_pred,
        'true_churn': y
    })
    out.to_csv(RESULTS_DIR / 'example_predictions.csv', index=False)

if __name__ == '__main__':
    main()
