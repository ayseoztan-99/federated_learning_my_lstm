import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Ağırlıkların başlatılması
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        # Bias değerlerinin başlatılması
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        # Gradyanlar için yer
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - x**2
    
    def forward(self, x, h_prev, c_prev):
        # Girdi ve gizli durumun birleştirilmesi
        combined = np.vstack((h_prev, x))
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        
        # Candidate memory cell
        c_hat = self.tanh(np.dot(self.Wc, combined) + self.bc)
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Memory cell
        c = f * c_prev + i * c_hat
        
        # Hidden state
        h = o * self.tanh(c)
        
        cache = (x, h_prev, c_prev, f, i, c_hat, o, c, combined)
        return h, c, cache
    
    def backward(self, dh, dc, cache):
        x, h_prev, c_prev, f, i, c_hat, o, c, combined = cache
        
        # Output gate gradyanları
        do = dh * self.tanh(c)
        do = do * self.sigmoid_derivative(o)
        self.dWo += np.dot(do, combined.T)
        self.dbo += do
        
        # Memory cell gradyanları
        dc += dh * o * self.tanh_derivative(self.tanh(c))
        
        # Forget gate gradyanları
        df = dc * c_prev
        df = df * self.sigmoid_derivative(f)
        self.dWf += np.dot(df, combined.T)
        self.dbf += df
        
        # Input gate gradyanları
        di = dc * c_hat
        di = di * self.sigmoid_derivative(i)
        self.dWi += np.dot(di, combined.T)
        self.dbi += di
        
        # Candidate memory cell gradyanları
        dc_hat = dc * i
        dc_hat = dc_hat * self.tanh_derivative(c_hat)
        self.dWc += np.dot(dc_hat, combined.T)
        self.dbc += dc_hat
        
        # Önceki hücre durumu için gradyanlar
        dc_prev = dc * f
        
        # Girdi ve önceki gizli durum için gradyanlar
        dcombined = (np.dot(self.Wf.T, df) + 
                    np.dot(self.Wi.T, di) + 
                    np.dot(self.Wc.T, dc_hat) + 
                    np.dot(self.Wo.T, do))
        
        dh_prev = dcombined[:self.hidden_size]
        dx = dcombined[self.hidden_size:]
        
        return dh_prev, dc_prev, dx
    
    def update_weights(self, learning_rate):
        # Ağırlıkları güncelle
        self.Wf -= learning_rate * self.dWf
        self.Wi -= learning_rate * self.dWi
        self.Wc -= learning_rate * self.dWc
        self.Wo -= learning_rate * self.dWo
        self.bf -= learning_rate * self.dbf
        self.bi -= learning_rate * self.dbi
        self.bc -= learning_rate * self.dbc
        self.bo -= learning_rate * self.dbo
        
        # Gradyanları sıfırla
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.dbo)

class CustomLSTMModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        self.dWy = np.zeros_like(self.Wy)
        self.dby = np.zeros_like(self.by)
        
    def forward(self, X):
        self.h = np.zeros((self.lstm.hidden_size, 1))
        self.c = np.zeros((self.lstm.hidden_size, 1))
        self.caches = []
        
        # Her zaman adımı için LSTM'i çalıştır
        for t in range(len(X)):
            x = X[t].reshape(-1, 1)
            self.h, self.c, cache = self.lstm.forward(x, self.h, self.c)
            self.caches.append(cache)
        
        # Son çıktıyı hesapla
        y = np.dot(self.Wy, self.h) + self.by
        return y
    
    def backward(self, dy, learning_rate):
        # Çıktı katmanı gradyanları
        self.dWy = np.dot(dy, self.h.T)
        self.dby = dy
        
        # LSTM'e geri yayılım
        dh = np.dot(self.Wy.T, dy)
        dc = np.zeros_like(self.c)
        
        # Zaman adımlarını tersine çevir
        for t in reversed(range(len(self.caches))):
            dh, dc, dx = self.lstm.backward(dh, dc, self.caches[t])
        
        # Ağırlıkları güncelle
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby
        self.lstm.update_weights(learning_rate)
    
    def mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def mse_loss_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size
    
    def mae_metric(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

class MultiLSTMModel:
    def __init__(self, recent_model, daily_model, weekly_model):
        self.recent_model = recent_model
        self.daily_model = daily_model
        self.weekly_model = weekly_model
        self.learning_rate = 0.001
        
    def forward(self, recent_data, daily_data, weekly_data):
        recent_output = self.recent_model.forward(recent_data)
        daily_output = self.daily_model.forward(daily_data)
        weekly_output = self.weekly_model.forward(weekly_data)
        
        # Çıktıları birleştir
        final_output = (recent_output + daily_output + weekly_output) / 3
        
        return {
            "output_recent": recent_output,
            "output_daily": daily_output,
            "output_weekly": weekly_output,
            "final_output": final_output
        }
    
    def compute_loss(self, outputs, targets):
        losses = {
            "output_recent": self.recent_model.mse_loss(outputs["output_recent"], targets["recent"]),
            "output_daily": self.daily_model.mse_loss(outputs["output_daily"], targets["daily"]),
            "output_weekly": self.weekly_model.mse_loss(outputs["output_weekly"], targets["weekly"]),
            "final_output": self.recent_model.mse_loss(outputs["final_output"], targets["final"])
        }
        
        # Kayıp ağırlıkları
        loss_weights = {
            "output_recent": 0.0,
            "output_daily": 0.0,
            "output_weekly": 0.0,
            "final_output": 1.0
        }
        
        # Toplam kayıp
        total_loss = sum(losses[key] * loss_weights[key] for key in losses)
        
        # MAE metriği
        mae = self.recent_model.mae_metric(outputs["final_output"], targets["final"])
        
        return total_loss, losses, mae
    
    def train_step(self, recent_data, daily_data, weekly_data, targets):
        # Forward pass
        outputs = self.forward(recent_data, daily_data, weekly_data)
        
        # Kayıp hesapla
        total_loss, losses, mae = self.compute_loss(outputs, targets)
        
        # Backward pass
        # Final çıktı için gradyan
        dy_final = self.recent_model.mse_loss_derivative(outputs["final_output"], targets["final"])
        
        # Her model için gradyanları hesapla ve güncelle
        self.recent_model.backward(dy_final / 3, self.learning_rate)
        self.daily_model.backward(dy_final / 3, self.learning_rate)
        self.weekly_model.backward(dy_final / 3, self.learning_rate)
        
        return total_loss, mae

def build_multi_lstm_model(th, td, tw, tp):
    # Her bir giriş için ayrı LSTM modelleri oluştur
    recent_model = CustomLSTMModel(1, 64, tp)
    daily_model = CustomLSTMModel(1, 32, tp)
    weekly_model = CustomLSTMModel(1, 32, tp)
    
    # Tüm modelleri birleştir
    return MultiLSTMModel(recent_model, daily_model, weekly_model)
