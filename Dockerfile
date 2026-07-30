# Menggunakan sistem operasi Python versi 3.9
FROM python:3.9

# Membuat folder kerja di dalam server
WORKDIR /code

# Memasukkan daftar library yang dibutuhkan
COPY ./requirements.txt /code/requirements.txt

# Meng-install semua library
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Memasukkan seluruh file kode kita ke dalam server
COPY . .

# Menyalakan API (Hugging Face mewajibkan penggunaan port 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]