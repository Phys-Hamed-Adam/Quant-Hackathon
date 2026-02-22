import utilities
import pandas as pd
import trading_model

if __name__ == "__main__":
    df = utilities.data_cleaning()
    model = trading_model.TradingModel(df)
    model.compile()
    print("Model compiled successfully!")
    model.plot_model()
    model.fit(df, epochs=1)  # Just a test run to check if the model can fit without errors
    print("Model fit successfully!")
    model.plot_training_history()

    