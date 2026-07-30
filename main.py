from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inisialisasi Aplikasi API
app = FastAPI(title="API Medical Checkup CDSS")

# 2. Pengaturan Keamanan (CORS) - Penting agar HTML bisa 'ngobrol' dengan API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan akses dari semua sumber (untuk tahap development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model ML yang sudah dilatih tadi
# Model ini sekarang jadi "otak" dari API kita
model = joblib.load("model_checkup.pkl")

# 4. Mendefinisikan Struktur Data Input
# Ini untuk memastikan data yang dikirim dari HTML sesuai dengan yang diminta model
class DataPasien(BaseModel):
    umur: int
    tensi_sistolik: int
    gula_puasa: int
    bmi: float

# 5. Membuat Jalur (Endpoint) untuk Menerima Data dan Mengembalikan Prediksi
@app.post("/prediksi")
def hitung_risiko(data: DataPasien):
    # Ubah data yang masuk menjadi format tabel (DataFrame) untuk model ML
    df = pd.DataFrame([data.model_dump()])
    
    # Minta model melakukan prediksi
    prediksi_kelas = model.predict(df)[0] # Hasilnya 0 (Rendah) atau 1 (Tinggi)
    probabilitas = model.predict_proba(df)[0] # Menghasilkan persentase
    
    # Ambil persentase kemungkinan risiko tinggi (indeks ke-1)
    persentase_risiko = round(probabilitas[1] * 100, 1)
    
    # Terjemahkan angka 0 atau 1 menjadi teks
    status = "Risiko Tinggi (Indikasi Sindrom Metabolik)" if prediksi_kelas == 1 else "Risiko Rendah"
    
    # Kembalikan jawaban ke website
    return {
        "status_risiko": status,
        "persentase": persentase_risiko,
        "pesan": "Prediksi berhasil dilakukan oleh model AI."
    }