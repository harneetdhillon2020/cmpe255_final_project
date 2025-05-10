import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def train_models_per_site(input_csv="../data/cleaned_all_sites.csv", output_dir="models"):
    df = pd.read_csv(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    for site in df['site'].unique():
        site_df = df[df['site'] == site].sort_values('date')
        site_df['lag1'] = site_df['elevation'].shift(1)
        site_df['lag2'] = site_df['elevation'].shift(2)
        site_df = site_df.dropna()

        X = site_df[['lag1', 'lag2']]
        y = site_df['elevation']

        split = int(len(site_df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        joblib.dump(model, f"{output_dir}/{site}_model.pkl")
        print(f"✅ Trained {site} | RMSE: {rmse:.2f} | Saved to {site}_model.pkl")
