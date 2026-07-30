import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Menyiapkan Dataset Dummy (Buatan)
# Anggap saja ini data rekam medis dari 10 pasien
data = {
    'umur': [25, 45, 62, 30, 50, 65, 22, 55, 70, 35],
    'tensi_sistolik': [110, 130, 150, 115, 140, 160, 105, 135, 155, 120],
    'gula_puasa': [90, 105, 140, 95, 120, 150, 85, 110, 145, 98],
    'bmi': [22.0, 26.5, 30.1, 23.5, 28.0, 31.5, 21.0, 27.5, 29.5, 24.0],
    # 0 = Sehat/Risiko Rendah, 1 = Risiko Tinggi (Sindrom Metabolik)
    'risiko': [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# 2. Memisahkan Input (X) dan Target/Output (y)
X = df[['umur', 'tensi_sistolik', 'gula_puasa', 'bmi']]
y = df['risiko']

# 3. Melatih Model (Mesin belajar mencari pola dari data di atas)
print("Mulai melatih model Machine Learning...")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("Model selesai dilatih!")

# 4. Menyimpan Model menjadi file 
joblib.dump(model, 'model_checkup.pkl')
print("Mantap! Model berhasil disimpan sebagai 'model_checkup.pkl'")