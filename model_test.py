import utilities
import pandas as pd
import numpy as np
import trading_model
import technical_indicators
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # To save the scaler for the hackathon evaluation script

def create_sequences(X, y, time_steps=10):
    """
    Converts 2D data into 3D sequences for Conv1D/GRU.
    Example: 10 days of history to predict the 11th day's label.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

if __name__ == "__main__":
    import os
    
    # Create the models folder if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # ----------------------------
    # Load and clean data
    # ----------------------------
    df = utilities.data_cleaning()                           
    df_spy, df_qqq = technical_indicators.prepare_data(df)  
    df_spy = utilities.add_price_direction_label(df_spy)    
    df_qqq = utilities.add_price_direction_label(df_qqq)

    # DANGER AVOIDANCE: Ensure raw prices are dropped before training!
    # Keep only your technical indicators
    raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_spy = df_spy.drop(columns=[col for col in raw_cols if col in df_spy.columns])
    df_qqq = df_qqq.drop(columns=[col for col in raw_cols if col in df_qqq.columns])

    # ----------------------------
    # Split features and labels
    # ----------------------------
    spy_X, spy_y = df_spy.drop(columns="Label"), df_spy["Label"]
    qqq_X, qqq_y = df_qqq.drop(columns="Label"), df_qqq["Label"]

    # CRITICAL FIX 1: shuffle=False for Time Series!
    spy_X_train, spy_X_test, spy_y_train, spy_y_test = train_test_split(
        spy_X, spy_y, test_size=0.2, shuffle=False
    )
    qqq_X_train, qqq_X_test, qqq_y_train, qqq_y_test = train_test_split(
        qqq_X, qqq_y, test_size=0.2, shuffle=False
    )

    # ----------------------------
    # CRITICAL FIX 2: Scale the Data
    # ----------------------------
    scaler_spy = StandardScaler()
    scaler_qqq = StandardScaler()

    # Fit ONLY on training data to prevent lookahead bias
    spy_X_train = pd.DataFrame(scaler_spy.fit_transform(spy_X_train), columns=spy_X.columns)
    spy_X_test = pd.DataFrame(scaler_spy.transform(spy_X_test), columns=spy_X.columns)
    
    qqq_X_train = pd.DataFrame(scaler_qqq.fit_transform(qqq_X_train), columns=qqq_X.columns)
    qqq_X_test = pd.DataFrame(scaler_qqq.transform(qqq_X_test), columns=qqq_X.columns)

    # Save scalers for strategy.py
    joblib.dump(scaler_spy, 'models/scaler_spy.pkl')
    joblib.dump(scaler_qqq, 'models/scaler_qqq.pkl')

    # ----------------------------
    # CRITICAL FIX 3: Convert to Sequences for GRU
    # ----------------------------
    timesteps = 10 # Model looks at 10 days of history
    
    spy_X_train_seq, spy_y_train_seq = create_sequences(spy_X_train, spy_y_train, timesteps)
    spy_X_test_seq, spy_y_test_seq   = create_sequences(spy_X_test, spy_y_test, timesteps)

    qqq_X_train_seq, qqq_y_train_seq = create_sequences(qqq_X_train, qqq_y_train, timesteps)
    qqq_X_test_seq, qqq_y_test_seq   = create_sequences(qqq_X_test, qqq_y_test, timesteps)

    features_spy = spy_X_train_seq.shape[2]
    features_qqq = qqq_X_train_seq.shape[2]

    # ----------------------------
    # Generate and compile models
    # ----------------------------
    models = {
        "SPY": trading_model.TradingModel(input_shape=(timesteps, features_spy)),
        "QQQ": trading_model.TradingModel(input_shape=(timesteps, features_qqq))
    }
    models["SPY"].compile()
    models["QQQ"].compile()

    print(f"Data ready! SPY Train Shape: {spy_X_train_seq.shape}")
    
    # ----------------------------
    # CRITICAL FIX 4: Convert One-Hot Labels to Binary Scalars
    # ----------------------------
    # This turns your labels from [0, 1] into a single 1 (or [1, 0] into a 0)
    if len(spy_y_train_seq.shape) > 1 and spy_y_train_seq.shape[1] > 1:
        spy_y_train_seq = np.argmax(spy_y_train_seq, axis=1)
        spy_y_test_seq  = np.argmax(spy_y_test_seq, axis=1)
        
    if len(qqq_y_train_seq.shape) > 1 and qqq_y_train_seq.shape[1] > 1:
        qqq_y_train_seq = np.argmax(qqq_y_train_seq, axis=1)
        qqq_y_test_seq  = np.argmax(qqq_y_test_seq, axis=1)

    # ----------------------------
    # Train models
    # ----------------------------
    print("Training SPY...")
    models["SPY"].fit(spy_X_train_seq, spy_y_train_seq, validation_data=(spy_X_test_seq, spy_y_test_seq), epochs=100, batch_size=32)
    
    print("Training QQQ...")
    models["QQQ"].fit(qqq_X_train_seq, qqq_y_train_seq, validation_data=(qqq_X_test_seq, qqq_y_test_seq), epochs=100, batch_size=32)
    
    import matplotlib.pyplot as plt

    # Get the raw probability predictions on your test set
    preds = models["SPY"].predict(spy_X_test_seq)

    # Plot a histogram
    plt.figure(figsize=(8, 5))
    plt.hist(preds, bins=50, color='blue', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='dashed', linewidth=2)
    plt.title("Distribution of SPY Model Predictions")
    plt.xlabel("Predicted Probability (0 = Down, 1 = Up)")
    plt.ylabel("Frequency")
    plt.show()

    # ----------------------------
    # Plot training history
    # ----------------------------
    models["SPY"].plot_training_history()
    models["QQQ"].plot_training_history()