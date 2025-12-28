import json
import pathlib

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.callbacks import EarlyStopping

from data_preprocessing import load_data, train_test_split_users
from sequence_generator import build_user_sequences
from lstm_model import build_lstm_model

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "mobile_money_logs.csv"
RESULTS_DIR = BASE_DIR / "results"
SEQ_LEN = 30

def main():
    df = load_data(str(DATA_PATH))
    train_df, test_df = train_test_split_users(df, test_size=0.2, random_state=42)
    X_train, y_train = build_user_sequences(train_df, seq_len=SEQ_LEN)
    X_test, y_test = build_user_sequences(test_df, seq_len=SEQ_LEN)

    model = build_lstm_model(seq_len=SEQ_LEN, num_features=X_train.shape[2], lr=1e-3)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    try:
        roc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        roc = None

    metrics = {'accuracy': acc, 'recall': rec, 'roc_auc': roc}
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    import numpy as np
    np.save(RESULTS_DIR / 'cm.npy', cm)
    np.save(RESULTS_DIR / 'roc_fpr.npy', fpr)
    np.save(RESULTS_DIR / 'roc_tpr.npy', tpr)

    model.save(RESULTS_DIR / 'lstm_model.h5')

if __name__ == '__main__':
    main()
