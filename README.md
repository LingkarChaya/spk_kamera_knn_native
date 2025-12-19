 # 📸 Sistem Klasifikasi Segmen Kamera (SPK - KNN Native)

Proyek ini adalah Sistem Pendukung Keputusan (SPK) untuk menentukan segmen kamera digital (**Entry Level, Mid Range, High End**) berdasarkan spesifikasi teknisnya.

Sistem ini dibangun menggunakan algoritma **K-Nearest Neighbors (KNN)** secara **Native** (tanpa library scikit-learn untuk proses prediksi) sesuai dengan studi kasus mata kuliah SPK.

## 🚀 Fitur Utama
* **KNN Native:** Algoritma klasifikasi dibangun dari nol menggunakan konsep OOP.
* **Normalisasi Min-Max:** Input data dinormalisasi agar fitur berat (gram) tidak mendominasi fitur zoom.
* **Web GUI:** Antarmuka pengguna interaktif menggunakan **Streamlit**.
* **Dataset:** Menggunakan data historis *1000 Cameras Dataset*.

## 📂 Struktur File
* `app.py`: File utama untuk menjalankan antarmuka web.
* `knn_native.py`: File *backend* berisi logika algoritma KNN dan Normalisasi.
* `data/`: Folder penyimpanan dataset CSV.

## 🛠️ Cara Menjalankan (Installation)

Pastikan Python sudah terinstall. Ikuti langkah berikut di terminal:

1.  **Clone Repository ini:**
    ```bash
    git clone [https://github.com/LingkarChaya/SPK_Kamera_KNN_Native.git](https://github.com/LingkarChaya/SPK_Kamera_KNN_Native.git)
    cd SPK_Kamera_KNN_Native
    ```

2.  **Install Library yang Dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Aplikasi:**
    ```bash
    streamlit run app.py
    ```

4.  Buka browser di alamat yang muncul (biasanya `http://localhost:8501`).

## 👨‍💻 Identitas Pengembang
* **Nama:** Lingkar Chaya C Hima
* **NIM:** 362358302178
* **Kelas:** 3E - TRPL
* **Kampus:** Politeknik Negeri Banyuwangi