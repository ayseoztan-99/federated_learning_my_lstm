# config.py

# Sunucu ve istemci ayarları
SERVER_ADDRESS = "localhost:8080"
NUM_ROUNDS = 1
LOCAL_EPOCHS = 1
MIN_FIT_CLIENTS = 10
MIN_EVAL_CLIENTS = 10
MIN_AVAILABLE_CLIENTS = 10
NUM_CLIENTS = 10  # Başlatılacak istemci sayısı
RESULTS_DIR = "results"
CLIENT_DATA_DIR = "client_datasets"

# Model hiperparametreleri
LEARNING_RATE = 0.001
HIDDEN_SIZE_RECENT = 64  # Recent model için gizli katman boyutu
HIDDEN_SIZE_DAILY = 32   # Daily model için gizli katman boyutu
HIDDEN_SIZE_WEEKLY = 32  # Weekly model için gizli katman boyutu

# Zaman penceresi parametreleri
TP = 12  # Tahmin penceresi (1 saat = 12 zaman adımı)
TH = 24  # Yakın geçmiş (son 2 saat = 24 zaman adımı)
TD = 12  # Günlük periyot (1 gün öncesinin aynı zaman dilimi = 12 zaman adımı)
TW = 24  # Haftalık periyot (1 hafta öncesinin aynı zaman dilimi = 24 zaman adımı)

# Veri ön işleme ayarları
SCALER_RANGE = (0, 1)  # MinMaxScaler için ölçeklendirme aralığı
BATCH_SIZE = 32        # Eğitim batch boyutu
