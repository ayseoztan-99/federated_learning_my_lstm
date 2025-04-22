# run_clients.py

import subprocess
import multiprocessing
import time
import os
from config import NUM_CLIENTS

def run_client(client_id):
    """İstemciyi başlat ve çalıştır"""
    print(f"[Client {client_id}] Başlatılıyor...")
    try:
        # İstemciyi başlat
        subprocess.run(["python", "client.py", str(client_id)], check=True)
        print(f"[Client {client_id}] Başarıyla tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"[Client {client_id}] Hata: {e}")
    except Exception as e:
        print(f"[Client {client_id}] Beklenmeyen hata: {e}")

def main():
    """Tüm istemcileri başlat"""
    print(f"[Server] {NUM_CLIENTS} istemci başlatılıyor...")
    
    # Sonuçlar dizinini oluştur
    os.makedirs("results", exist_ok=True)
    
    # İstemcileri başlat
    processes = []
    for i in range(1, NUM_CLIENTS + 1):
        p = multiprocessing.Process(target=run_client, args=(i,))
        p.start()
        processes.append(p)
        time.sleep(1)  # İstemciler arasında 1 saniye bekle
    
    # Tüm istemcilerin tamamlanmasını bekle
    for p in processes:
        p.join()
    
    print("[Server] Tüm istemci işlemleri tamamlandı.")

if __name__ == "__main__":
    main()
