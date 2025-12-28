from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_rnn_model(seq_len: int, num_features: int, lr: float = 1e-3):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(seq_len, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(seq_len: int, num_features: int, lr: float = 1e-3):
    model = Sequential()
    model.add(GRU(64, input_shape=(seq_len, num_features)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model
