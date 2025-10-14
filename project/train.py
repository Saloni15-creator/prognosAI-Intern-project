import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_data(data_dir):
    sequences = np.load(os.path.join(data_dir, 'sequences.npy'))
    metadata = np.genfromtxt(os.path.join(data_dir, 'metadata.csv'), delimiter=',', names=True, dtype=None, encoding=None)
    rul = np.array([row['RUL'] for row in metadata])
    print(f"Loaded {data_dir} data: {sequences.shape}, RULs: {rul.shape}")
    return sequences, rul

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    return model

def train_model(train_x, train_y, val_x, val_y, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    model = build_model((train_x.shape[1], train_x.shape[2]))

    checkpoint_cb = callbacks.ModelCheckpoint(
        os.path.join(save_dir, 'best_model.keras'),
        monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )

    earlystop_cb = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )

    model.save(os.path.join(save_dir, 'final_model.keras'))
    return model, history

def evaluate_model(model, test_x, test_y, save_dir='models'):
    preds = model.predict(test_x).flatten()

    mse = mean_squared_error(test_y, preds)
    mae = mean_absolute_error(test_y, preds)
    r2 = r2_score(test_y, preds)

    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'R2 Score: {r2:.4f}\n')

    print("\nModel Evaluation:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, mae, r2

def main():
    # Load preprocessed train and test data
    train_dir = 'processed_data/train'
    test_dir = 'processed_data/test'

    train_x, train_y = load_data(train_dir)
    test_x, test_y = load_data(test_dir)

    # Split validation set
    val_split = int(0.2 * len(train_x))
    val_x, val_y = train_x[:val_split], train_y[:val_split]
    train_x, train_y = train_x[val_split:], train_y[val_split:]

    # Train model
    model, history = train_model(train_x, train_y, val_x, val_y, save_dir='models')

    # Evaluate model
    evaluate_model(model, test_x, test_y, save_dir='models')

if __name__ == '__main__':
    main()