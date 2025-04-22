import flwr as fl
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from config import SERVER_ADDRESS, CLIENT_DATA_DIR, TH, TD, TW, TP, LOCAL_EPOCHS
from model import build_multi_lstm_model

import tensorflow as tf

import tensorflow as tf
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"[Client {client_id}] Initializing...")

file_path = os.path.join(CLIENT_DATA_DIR, f"client_data_{client_id}.csv")
df = pd.read_csv(file_path)
df = df.sort_values(by=["location", "timestep"]).reset_index(drop=True)

scaler = MinMaxScaler()
df["flow"] = scaler.fit_transform(df["flow"].values.reshape(-1, 1))

timesteps_per_day = 288
train_days = 50
test_days = 12
train_size = train_days * timesteps_per_day

X_recent, X_daily, X_weekly, Y = [], [], [], []
locations = df["location"].unique()

for loc in locations:
    df_loc = df[df["location"] == loc].reset_index(drop=True)
    for i in range(train_size, len(df_loc) - TP):
        recent = df_loc["flow"].iloc[i - TH:i].values
        daily = df_loc["flow"].iloc[i - TD - timesteps_per_day:i - timesteps_per_day].values
        weekly = df_loc["flow"].iloc[i - TW - 7 * timesteps_per_day:i - 7 * timesteps_per_day].values
        target = df_loc["flow"].iloc[i:i + TP].values

        if len(recent) == TH and len(daily) == TD and len(weekly) == TW and len(target) == TP:
            X_recent.append(recent)
            X_daily.append(daily)
            X_weekly.append(weekly)
            Y.append(target)

X_recent = np.array(X_recent)
X_daily = np.array(X_daily)
X_weekly = np.array(X_weekly)
Y = np.array(Y)

split = int(0.8 * len(Y))
x_recent_train, x_recent_test = X_recent[:split], X_recent[split:]
x_daily_train, x_daily_test = X_daily[:split], X_daily[split:]
x_weekly_train, x_weekly_test = X_weekly[:split], X_weekly[split:]
y_train, y_test = Y[:split], Y[split:]

model = build_multi_lstm_model(TH, TD, TW, TP)
print(f"[Client {client_id}] Model has been built.")

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Model ağırlıklarını al
        weights = []
        for model_name in ['recent', 'daily', 'weekly']:
            model_part = getattr(model, f'{model_name}_model')
            weights.extend([
                model_part.lstm.Wf,
                model_part.lstm.Wi,
                model_part.lstm.Wc,
                model_part.lstm.Wo,
                model_part.lstm.bf,
                model_part.lstm.bi,
                model_part.lstm.bc,
                model_part.lstm.bo,
                model_part.Wy,
                model_part.by
            ])
        return weights

    def set_parameters(self, parameters):
        # Model ağırlıklarını ayarla
        idx = 0
        for model_name in ['recent', 'daily', 'weekly']:
            model_part = getattr(model, f'{model_name}_model')
            model_part.lstm.Wf = parameters[idx]
            model_part.lstm.Wi = parameters[idx + 1]
            model_part.lstm.Wc = parameters[idx + 2]
            model_part.lstm.Wo = parameters[idx + 3]
            model_part.lstm.bf = parameters[idx + 4]
            model_part.lstm.bi = parameters[idx + 5]
            model_part.lstm.bc = parameters[idx + 6]
            model_part.lstm.bo = parameters[idx + 7]
            model_part.Wy = parameters[idx + 8]
            model_part.by = parameters[idx + 9]
            idx += 10

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Eğitim döngüsü
        for epoch in range(LOCAL_EPOCHS):
            total_loss = 0
            for i in range(len(x_recent_train)):
                # Hedef değerler
                targets = {
                    "recent": y_train[i].reshape(-1, 1),
                    "daily": y_train[i].reshape(-1, 1),
                    "weekly": y_train[i].reshape(-1, 1),
                    "final": y_train[i].reshape(-1, 1)
                }
                
                # Eğitim adımı
                loss, mae = model.train_step(
                    x_recent_train[i],
                    x_daily_train[i],
                    x_weekly_train[i],
                    targets
                )
                total_loss += loss
            
            avg_loss = total_loss / len(x_recent_train)
            print(f"[Client {client_id}] Epoch {epoch + 1}/{LOCAL_EPOCHS}, Loss: {avg_loss:.4f}")
        
        return self.get_parameters(config), len(x_recent_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        total_loss = 0
        predictions = []
        true_values = []
        
        for i in range(len(x_recent_test)):
            # Test verisi için forward pass
            outputs = model.forward(
                x_recent_test[i],
                x_daily_test[i],
                x_weekly_test[i]
            )
            
            # Kayıp hesapla
            targets = {
                "recent": y_test[i].reshape(-1, 1),
                "daily": y_test[i].reshape(-1, 1),
                "weekly": y_test[i].reshape(-1, 1),
                "final": y_test[i].reshape(-1, 1)
            }
            loss, _, _ = model.compute_loss(outputs, targets)
            total_loss += loss
            
            # Tahminleri ve gerçek değerleri topla
            predictions.append(outputs["final_output"].flatten())
            true_values.append(y_test[i])
        
        # Metrikleri hesapla
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        pred_final = scaler.inverse_transform(predictions)
        true_final = scaler.inverse_transform(true_values)
        
        rmse = np.sqrt(mean_squared_error(true_final, pred_final))
        r2 = r2_score(true_final, pred_final)
        mae = np.mean(np.abs(true_final - pred_final))
        mape = np.mean(np.abs((true_final - pred_final) / (true_final + 1e-5))) * 100
        
        avg_loss = total_loss / len(x_recent_test)
        print(f"[Client {client_id}] Loss: {avg_loss:.4f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        
        return avg_loss, len(x_recent_test), {
            "rmse": float(rmse),
            "r2": float(r2),
            "mae": float(mae),
            "mape": float(mape),
            "client_id": client_id
        }

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FLClient())
