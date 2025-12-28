import json
import pathlib

from sklearn.metrics import recall_score
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_preprocessing import load_data, train_test_split_users
from sequence_generator import build_user_sequences

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mobile_money_logs.csv"
RESULTS_DIR = BASE_DIR / "results"
SEQ_LEN = 30

def main():
    df = load_data(str(DATA_PATH))
    train_df, val_df = train_test_split_users(df, test_size=0.2, random_state=123)
    X_train, y_train = build_user_sequences(train_df, seq_len=SEQ_LEN)
    X_val, y_val = build_user_sequences(val_df, seq_len=SEQ_LEN)

    configs = [
        {'lr': 1e-3, 'units': 64},
        {'lr': 5e-4, 'units': 128},
    ]
    results = []

    for cfg in configs:
        model = Sequential()
        model.add(LSTM(cfg['units'], input_shape=(SEQ_LEN, X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=cfg['lr']), loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        y_prob = model.predict(X_val).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        rec = float(recall_score(y_val, y_pred))
        out_cfg = dict(cfg)
        out_cfg['recall'] = rec
        results.append(out_cfg)

    (RESULTS_DIR / 'hyperparam_results.json').write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
